import argparse
import os
import pickle
from builtins import print
from pathlib import Path

import cv2
import imageio
# import mmcv
import mmengine
import numpy as np
import torch
from tqdm import tqdm, trange

from lib import utils, temporalpoints, tineuvox
from lib.load_data import load_data


def create_empty_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            os.remove(os.path.join(path, filename))


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument("--seed", type=int, default=0, help='Random seed')
    # # testing options
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--degree_threshold", type=float, default=0.)
    parser.add_argument("--skip_load_images", action='store_true')
    parser.add_argument('--num_fps', default=-1, type=int)
    return parser


@torch.no_grad()
def render_viewpoints(
    model, render_poses, HW, Ks, ndc, render_kwargs,
    gt_imgs=None, savedir=None, test_times=None, render_factor=0,
    inverse_y=False, flip_x=False, flip_y=False, batch_size=4096 * 2, verbose=True,
    render_pcd_direct=False, render_flow=False,
    fixed_viewdirs=None
):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    print(render_poses.shape, HW.shape, Ks.shape)
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = torch.clone(Ks)
        HW = HW // render_factor
        Ks[:, :2, :3] = Ks[:, :2, :3] // render_factor

    rgbs = []
    depths = []
    weights = []
    flows = []
    psnrs = []
    ssims = []
    joints = {}
    bones = None
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses, disable=not verbose)):
        H, W = HW[i]
        K = Ks[i].to(torch.float32)
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)

        if fixed_viewdirs is not None:
            viewdirs = fixed_viewdirs

        pixel_coords = torch.stack(torch.meshgrid(torch.arange(0, W), torch.arange(0, H)), dim=-1)
        pixel_coords = pixel_coords.reshape(-1, 2).to(torch.float32).to(rays_o.device)

        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        time_one = test_times[i] * torch.ones_like(rays_o[:, 0:1])

        if type(model) is not temporalpoints.TemporalPoints:
            keys = ['rgb_marched', 'depth']
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, ts, **render_kwargs).items() if k in keys}
                for ro, rd, vd, ts in zip(rays_o.split(batch_size, 0),
                    rays_d.split(batch_size, 0),
                    viewdirs.split(batch_size, 0),
                    time_one.split(batch_size, 0))
            ]
        else:
            keys = ['rgb_marched', 'depth', 'weights']
            if render_flow: keys.append('flow')
            render_result_chunks = []

            for ro, rd, vd, ts, px in zip(rays_o.split(batch_size, 0),
                rays_d.split(batch_size, 0),
                viewdirs.split(batch_size, 0),
                time_one.split(batch_size, 0),
                pixel_coords.split(batch_size, 0)):
                render_kwargs['rays_o'] = ro
                render_kwargs['rays_d'] = rd
                render_kwargs['viewdirs'] = vd
                render_kwargs['pixel_coords'] = px
                cam_per_ray = torch.zeros(len(ro))[:, None]

                if render_flow:
                    i_delta = max(i - 1, 0)
                    flow_t_delta = test_times[i_delta] - test_times[i]
                else:
                    flow_t_delta = None

                out = model(ts[0], render_depth=True, render_kwargs=render_kwargs, render_weights=True,
                    render_pcd_direct=render_pcd_direct, poses=c2w[None], Ks=Ks[i][None],
                    cam_per_ray=cam_per_ray, get_skeleton=True)

                if out['joints'] is not None:
                    if not render_kwargs['inverse_y']:
                        out['joints'][:, :, 0] = (HW[0, 0] - 1) - out['joints'][:, :, 0]

                    if not i in joints.keys():
                        joints[i] = out['joints'][0].cpu().numpy()
                        bones = out['bones']

                if render_pcd_direct:
                    out['rgb_marched'] = out['rgb_marched_direct']

                chunk = {k: v for k, v in out.items() if k in keys}
                render_result_chunks.append(chunk)

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        try:
            weight = render_result['weights'].cpu().numpy()
            weights.append(weight)
        except:
            pass

        rgbs.append(rgb)
        depths.append(depth)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        # create text file and write results into a single file
        if savedir is not None:
            with open(os.path.join(savedir, 'results.txt'), 'w') as f:
                f.write('psnr: ' + str(np.mean(psnrs)) + '\n')
                f.write('ssim: ' + str(np.mean(ssims)) + '\n')
                f.write('lpips_alex: ' + str(np.mean(lpips_alex)) + '\n')
                f.write('lpips_vgg: ' + str(np.mean(lpips_vgg)) + '\n')

        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ssim', np.mean(ssims), '(avg)')
        print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        create_empty_dir(os.path.join(savedir, 'images'))
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, 'images', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

        create_empty_dir(os.path.join(savedir, 'gt'))
        for i in trange(len(gt_imgs)):
            rgb8 = utils.to8b(gt_imgs[i])
            filename = os.path.join(savedir, 'gt', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

        # for i in trange(len(weights)):
        #     rgb8 = utils.to8b(weights[i])
        #     filename = os.path.join(savedir, 'weights_{:03d}.png'.format(i))
        #     imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    weights = np.array(weights)
    flows = np.array(flows)
    joints = [joints[i] for i in range(len(joints))]
    joints = np.array(joints).astype(np.int32)

    for i in range(len(weights)):
        img = weights[i]

        for bone in bones:
            img = cv2.line(img, joints[i][bone[0]], joints[i][bone[1]], color=(0, 0, 0), thickness=1)

        for j in range(joints.shape[1]):
            img = cv2.circle(img, joints[i][j], radius=3, color=(0, 0, 0), thickness=-1)

        weights[i] = img
    if savedir is not None:
        print(f'Writing images to {savedir}')
        create_empty_dir(os.path.join(savedir, 'bone'))
        for i in trange(len(weights)):
            rgb8 = utils.to8b(weights[i])
            filename = os.path.join(savedir, 'bone', '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


@torch.no_grad()
def render_speed(
    model, render_poses, HW, Ks, ndc, render_kwargs,
    savedir=None, test_times=None, render_factor=0,
    inverse_y=False, flip_x=False, flip_y=False, batch_size=4096 * 2, verbose=True,
    render_pcd_direct=False, render_flow=False,
    fixed_viewdirs=None, num_images=-1,
):
    if render_factor != 0:
        HW = np.copy(HW)
        Ks = torch.clone(Ks)
        HW = HW // render_factor
        Ks[:, :2, :3] = Ks[:, :2, :3] // render_factor

    if num_images > 0:
        render_poses_, HW_, Ks_, test_times_ = [], [], [], []
        for i in range(len(render_poses)):
            start = i * num_images // len(render_poses)
            end = (i + 1) * num_images // len(render_poses)
            render_poses_.extend(render_poses[i] for k in range(start, end))
            HW_.extend(HW[i] for k in range(start, end))
            Ks_.extend(Ks[i] for k in range(start, end))
            test_times_.extend((k - start) / (end - start) for k in range(start, end))
        render_poses = torch.stack(render_poses_, dim=0)
        HW = np.stack(HW_, axis=0)
        Ks = torch.stack(Ks_, dim=0)
        test_times = test_times.new_tensor(test_times_)
        assert len(render_poses) == num_images

    times = []
    for i, c2w in enumerate(tqdm(render_poses, disable=not verbose)):
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        H, W = HW[i]
        K = Ks[i].to(torch.float32)
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)

        if fixed_viewdirs is not None:
            viewdirs = fixed_viewdirs

        pixel_coords = torch.stack(torch.meshgrid(torch.arange(0, W), torch.arange(0, H)), dim=-1)
        pixel_coords = pixel_coords.reshape(-1, 2).to(torch.float32).to(rays_o.device)

        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        time_one = test_times[i] * torch.ones_like(rays_o[:, 0:1])

        if type(model) is not temporalpoints.TemporalPoints:
            keys = ['rgb_marched', 'depth']
            render_result_chunks = [
                {k: v for k, v in model(ro, rd, vd, ts, **render_kwargs).items() if k in keys}
                for ro, rd, vd, ts in zip(rays_o.split(batch_size, 0),
                    rays_d.split(batch_size, 0),
                    viewdirs.split(batch_size, 0),
                    time_one.split(batch_size, 0))
            ]
        else:
            keys = ['rgb_marched', 'depth', 'weights']
            if render_flow: keys.append('flow')
            render_result_chunks = []

            for ro, rd, vd, ts, px in zip(rays_o.split(batch_size, 0),
                rays_d.split(batch_size, 0),
                viewdirs.split(batch_size, 0),
                time_one.split(batch_size, 0),
                pixel_coords.split(batch_size, 0)):
                render_kwargs['rays_o'] = ro
                render_kwargs['rays_d'] = rd
                render_kwargs['viewdirs'] = vd
                render_kwargs['pixel_coords'] = px
                cam_per_ray = torch.zeros(len(ro))[:, None]

                if render_flow:
                    i_delta = max(i - 1, 0)
                    flow_t_delta = test_times[i_delta] - test_times[i]
                else:
                    flow_t_delta = None

                out = model(ts[0], render_depth=True, render_kwargs=render_kwargs, render_weights=True,
                    render_pcd_direct=render_pcd_direct, poses=c2w[None], Ks=Ks[i][None],
                    cam_per_ray=cam_per_ray, get_skeleton=True)

                if out['joints'] is not None:
                    if not render_kwargs['inverse_y']:
                        out['joints'][:, :, 0] = (HW[0, 0] - 1) - out['joints'][:, :, 0]

                if render_pcd_direct:
                    out['rgb_marched'] = out['rgb_marched_direct']

                chunk = {k: v for k, v in out.items() if k in keys}
                render_result_chunks.append(chunk)

        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched']
        depth = render_result['depth']

        end_time = torch.cuda.Event(enable_timing=True)
        end_time.record()
        start_time.synchronize()
        end_time.synchronize()
        times.append(start_time.elapsed_time(end_time))
    all_times = sum(times) / 1000
    fps = len(times) / all_times
    print(f"Render {len(times)} images in {all_times:.6f} s, FPS: {fps:.2f}.")
    with open(os.path.join(savedir, 'results.txt'), 'a') as f:
        f.write('\n')
        f.write(f'num_images: {len(times)}\n')
        f.write(f'time(s): {all_times:.6f}\n')
        f.write(f'FPS: {fps:.6f}\n')


def load_everything(args, cfg, use_cache=False, overwrite=False):
    '''Load images / poses / camera settings / data split.
    '''
    cfg.data.skip_images = bool(args.skip_load_images)

    if not os.path.isdir(cfg.data.datadir):
        cache_file_folder = cfg.data.datadir.split('.pickle')[0]
        os.makedirs(cfg.data.datadir.split('.pickle')[0], exist_ok=True)
        cache_file = Path(cache_file_folder) / 'cache.pth'
    else:
        cache_file = Path(cfg.data.datadir) / 'cache.pth'
    if use_cache and not overwrite and cache_file.is_file():
        with cache_file.open("rb") as f:
            data_dict = pickle.load(f)
        return data_dict

    try:
        bg_col = cfg.train_config.bg_col
    except:
        bg_col = None
    data_dict = load_data(cfg.data, cfg, True, bg_col=bg_col)
    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images', 'times', 'render_times',
        'img_to_cam', 'masks'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    if use_cache:
        with cache_file.open('wb') as f:
            pickle.dump(data_dict, f)

    return data_dict


def test():
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    data_dict = None

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg, use_cache=args.use_cache, overwrite=args.overwrite_cache)
    read_path: str = os.path.join(cfg.basedir, cfg.expname)  # noqa
    save_path = read_path

    # load model for rendering
    # cfg.basedir += args.basedir_append_suffix
    ckpt_path = os.path.join(save_path, 'temporalpoints_last.tar')
    model_class = temporalpoints.TemporalPoints

    model = utils.load_model(model_class, ckpt_path).to(device)
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    near = data_dict['near']
    far = data_dict['far']
    stepsize = cfg.model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
        'render_kwargs': {
            'near': near,
            'far': far,
            'bg': cfg.train_config.bg_col,
            'stepsize': stepsize,
            'render_depth': True,
            'inverse_y': cfg.data.inverse_y,
        },
    }

    if args.degree_threshold > 0:
        times = data_dict['times'].unique().unsqueeze(-1)
        joints, bones, new_joints, new_bones, prune_bones, _, _, res = model.simplify_skeleton(
            times,
            deg_threshold=args.degree_threshold,
            five_percent_heuristic=True,
            visualise_canonical=False)  # If visualise canonical, we will overwrite the bones
    else:
        prune_bones = torch.tensor([])

    # render test-set and eval
    testsavedir = os.path.join(save_path, f'test_{ckpt_name}')
    os.makedirs(testsavedir, exist_ok=True)

    # save threshold and static joints in txt file
    with open(os.path.join(testsavedir, 'threshold.txt'), 'w') as f:
        f.write(f'{args.degree_threshold}\n')
        f.write(f'Static joints: {prune_bones.sum()} / {len(prune_bones)}')

    for k, v in data_dict.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            print(k, v.shape)
        else:
            print(k, type(v))
    # return
    render_viewpoints(
        render_poses=data_dict['poses'][data_dict['i_test']],
        HW=data_dict['HW'][data_dict['i_test']],
        Ks=data_dict['Ks'][data_dict['img_to_cam'][data_dict['i_test']]],
        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
        savedir=testsavedir,
        test_times=data_dict['times'][data_dict['i_test']],
        **render_viewpoints_kwargs
    )

    render_speed(
        render_poses=data_dict['poses'][data_dict['i_test']],
        HW=data_dict['HW'][data_dict['i_test']],
        Ks=data_dict['Ks'][data_dict['img_to_cam'][data_dict['i_test']]],
        savedir=testsavedir,
        test_times=data_dict['times'][data_dict['i_test']],
        num_images=args.num_fps,
        **render_viewpoints_kwargs
    )


if __name__ == '__main__':
    test()
