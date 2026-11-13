#
# SG-3DGS: Semantic-Guided 3D Gaussian Splatting Training Script
#
# Integrates three innovations:
#   1. SA-ADC  - Semantic-Aware Adaptive Density Control
#   2. MFG-Loss - Multi-Granularity Feature-Guided Loss
#   3. SP-Reg  - Structure-Preserving Gaussian Regularization
#
# Based on the original 3DGS training script by Inria GRAPHDECO.
#

import os
import sys
import torch
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# SG-3DGS modules
from sg3dgs.region_manager import RegionManager, project_gaussians_to_2d, REGION_TEXTURED_SURFACE
from sg3dgs.attention_manager import AttentionManager
from sg3dgs.sa_adc import SemanticAwareDensifier
from sg3dgs.mfg_loss import MFGLoss
from sg3dgs.sp_reg import StructurePreservingRegularizer

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def load_sg3dgs_config(config_path: str = None) -> dict:
    """Load SG-3DGS hyperparameter config from YAML, with defaults."""
    defaults = {
        # SA-ADC parameters
        "sa_adc": {
            "enabled": True,
            "alpha_linear": 0.5,
            "alpha_edge": 0.7,
            "alpha_textured": 1.0,
            "alpha_flat": 1.5,
            "boundary_threshold_mult": 0.3,
            "boundary_densify_interval": 500,
            "sa_adc_start_iter": 3000,
            "cache_update_interval": 100,
        },
        # MFG-Loss parameters
        "mfg_loss": {
            "enabled": True,
            "lambda_dssim": 0.2,
            "edge_loss_weight": 0.05,
            "edge_loss_start_iter": 3000,
            "use_edge_loss": True,
            "freq_loss_weight": 0.01,
            "freq_loss_start_iter": 5000,
            "freq_attention_threshold": 0.7,
        },
        # SP-Reg parameters
        "sp_reg": {
            "enabled": True,
            "lambda_aniso": 0.01,
            "aniso_start_iter": 5000,
            "linear_aspect_target": 3.0,
            "surface_flatness_target": 0.3,
            "lambda_opacity": 0.005,
            "opacity_start_iter": 15000,
            "lambda_boundary": 0.005,
            "boundary_start_iter": 5000,
            "lambda_scale": 0.001,
            "scale_start_iter": 3000,
            "max_scale_ratio": 10.0,
        },
        # Data paths
        "semantic_folder": "semantic",
        "features_folder": "features",
    }

    if config_path and os.path.exists(config_path) and YAML_AVAILABLE:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            for section in ['sa_adc', 'mfg_loss', 'sp_reg']:
                if section in user_cfg:
                    defaults[section].update(user_cfg[section])
            for key in ['semantic_folder', 'features_folder']:
                if key in user_cfg:
                    defaults[key] = user_cfg[key]

    return defaults


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, sg3dgs_config):
    """
    SG-3DGS training loop with semantic-guided innovations.
    """

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # ========================================================================
    # SG-3DGS: Initialize semantic modules
    # ========================================================================
    cfg = sg3dgs_config

    # --- Region Manager (SA-ADC + SP-Reg) ---
    region_manager = RegionManager(dataset.source_path, cfg["semantic_folder"])
    region_loaded = region_manager.load()
    if not region_loaded:
        print("[SG-3DGS] WARNING: No region maps found. SA-ADC and SP-Reg boundary constraints will be limited.")

    # --- Attention Manager (MFG-Loss) ---
    attention_manager = AttentionManager(dataset.source_path, cfg["features_folder"])
    attention_loaded = attention_manager.load()
    if not attention_loaded:
        print("[SG-3DGS] WARNING: No attention maps found. MFG-Loss will use uniform weights.")

    # --- SA-ADC ---
    sa_adc_enabled = cfg["sa_adc"]["enabled"] and region_loaded
    densifier = SemanticAwareDensifier(**{k: v for k, v in cfg["sa_adc"].items() if k != "enabled"})
    if sa_adc_enabled:
        print("[SG-3DGS] SA-ADC: ENABLED")
    else:
        print("[SG-3DGS] SA-ADC: DISABLED (no region maps or config)")

    # --- MFG-Loss ---
    mfg_enabled = cfg["mfg_loss"]["enabled"]
    mfg_loss = MFGLoss(**{k: v for k, v in cfg["mfg_loss"].items() if k != "enabled"})
    print(f"[SG-3DGS] MFG-Loss: {'ENABLED' if mfg_enabled else 'DISABLED'}")

    # --- SP-Reg ---
    sp_reg_enabled = cfg["sp_reg"]["enabled"]
    sp_reg = StructurePreservingRegularizer(**{k: v for k, v in cfg["sp_reg"].items() if k != "enabled"})
    print(f"[SG-3DGS] SP-Reg: {'ENABLED' if sp_reg_enabled else 'DISABLED'}")

    # ========================================================================
    # Training loop
    # ========================================================================
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_reg_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="SG-3DGS Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # ==================================================================
        # SG-3DGS: Compute loss with MFG-Loss
        # ==================================================================
        loss_dict = None
        reg_losses = None
        if mfg_enabled:
            # Get attention map for this image
            img_name = viewpoint_cam.image_name
            attention_map = attention_manager.get_attention_map(
                img_name, target_h=gt_image.shape[1], target_w=gt_image.shape[2]
            )

            fused_fn = fused_ssim if FUSED_SSIM_AVAILABLE else None
            loss_dict = mfg_loss.compute_loss(
                image, gt_image, attention_map,
                ssim_fn=ssim, iteration=iteration,
                fused_ssim_fn=fused_fn
            )
            loss = loss_dict['total']
            Ll1 = loss_dict['weighted_l1']
        else:
            # Standard loss (original 3DGS)
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization (from original)
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # ==================================================================
        # SG-3DGS: SP-Reg regularization losses
        # ==================================================================
        reg_loss_val = 0.0
        if sp_reg_enabled:
            # Get region types for current Gaussians
            region_types = None
            boundary_proximity = None

            # Try SA-ADC cache first (most efficient)
            if sa_adc_enabled and densifier._cached_region_types is not None:
                rt = densifier._cached_region_types
                bp = densifier._cached_boundary_proximity
                if rt.shape[0] == gaussians.get_xyz.shape[0]:
                    region_types = rt
                    boundary_proximity = bp

            # Fallback: project directly if region maps available but SA-ADC disabled
            if region_types is None and region_loaded and iteration % 100 == 0:
                region_map = region_manager.get_region_map(viewpoint_cam.image_name)
                boundary_map = region_manager.get_boundary_map(viewpoint_cam.image_name)
                if region_map is not None:
                    with torch.no_grad():
                        pos2d, valid = project_gaussians_to_2d(gaussians.get_xyz, viewpoint_cam)
                        N = gaussians.get_xyz.shape[0]
                        region_types = torch.full((N,), REGION_TEXTURED_SURFACE, device="cuda", dtype=torch.long)
                        boundary_proximity = torch.zeros(N, device="cuda")
                        if valid.any():
                            H, W = region_map.shape
                            x = pos2d[valid, 0].long().clamp(0, W - 1)
                            y = pos2d[valid, 1].long().clamp(0, H - 1)
                            region_types[valid] = region_map[y, x]
                            if boundary_map is not None:
                                boundary_proximity[valid] = boundary_map[y, x]

            reg_losses = sp_reg.compute_total_loss(
                gaussians, iteration,
                region_types=region_types,
                boundary_proximity=boundary_proximity
            )
            reg_total = reg_losses['total']
            if reg_total.item() > 0:
                loss = loss + reg_total
                reg_loss_val = reg_total.item()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_reg_for_log = 0.4 * reg_loss_val + 0.6 * ema_reg_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth": f"{ema_Ll1depth_for_log:.{7}f}",
                    "Reg": f"{ema_reg_for_log:.{5}f}",
                    "#GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # SG-3DGS specific Tensorboard logging
            if tb_writer and iteration % 100 == 0:
                if loss_dict is not None:
                    for k, v in loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            tb_writer.add_scalar(f'sg3dgs_mfg/{k}', v.item(), iteration)
                if reg_losses is not None:
                    for k, v in reg_losses.items():
                        if isinstance(v, torch.Tensor):
                            tb_writer.add_scalar(f'sg3dgs_spreg/{k}', v.item(), iteration)
                tb_writer.add_scalar('sg3dgs/reg_total', reg_loss_val, iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ==============================================================
            # SG-3DGS: Semantic-Aware Densification
            # ==============================================================
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # Boundary-enhanced densification BEFORE regular densify
                    # (uses accumulated gradients; we save/restore them for regular densify)
                    if sa_adc_enabled and densifier.should_boundary_densify(iteration):
                        region_map = region_manager.get_region_map(viewpoint_cam.image_name)
                        boundary_map = region_manager.get_boundary_map(viewpoint_cam.image_name)
                        densifier.update_cache(gaussians, viewpoint_cam, region_map, boundary_map, iteration)

                        # Save gradient state before boundary clone (densification_postfix resets them)
                        saved_grad_accum = gaussians.xyz_gradient_accum.clone()
                        saved_denom = gaussians.denom.clone()
                        saved_max_radii = gaussians.max_radii2D.clone()

                        gaussians.tmp_radii = radii
                        clone_mask = densifier.get_boundary_clone_mask(
                            gaussians, opt.densify_grad_threshold)
                        if clone_mask.any():
                            gaussians.boundary_clone(clone_mask)
                            # Restore gradient state for original Gaussians,
                            # new clones get zeros (appended by densification_postfix)
                            n_orig = saved_grad_accum.shape[0]
                            gaussians.xyz_gradient_accum[:n_orig] = saved_grad_accum
                            gaussians.denom[:n_orig] = saved_denom
                            gaussians.max_radii2D[:n_orig] = saved_max_radii
                            if iteration % 1000 == 0:
                                print(f"  [SA-ADC] Boundary clone: {clone_mask.sum().item()} Gaussians")
                        gaussians.tmp_radii = None

                    if sa_adc_enabled and densifier.is_active(iteration):
                        # Force cache refresh to match current Gaussian count
                        densifier._cache_iter = -1
                        region_map = region_manager.get_region_map(viewpoint_cam.image_name)
                        boundary_map = region_manager.get_boundary_map(viewpoint_cam.image_name)
                        densifier.update_cache(gaussians, viewpoint_cam, region_map, boundary_map, iteration)

                        # Compute adaptive thresholds
                        N = gaussians.get_xyz.shape[0]
                        adaptive_thresholds = densifier.compute_adaptive_threshold(
                            opt.densify_grad_threshold, N)

                        # Semantic-aware densification
                        gaussians.densify_and_prune_semantic(
                            opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                            size_threshold, radii, adaptive_thresholds=adaptive_thresholds)
                    else:
                        # Original densification
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Periodic SA-ADC cache refresh (even outside densification, for SP-Reg)
            if sa_adc_enabled and iteration % densifier.cache_update_interval == 0:
                region_map = region_manager.get_region_map(viewpoint_cam.image_name)
                boundary_map = region_manager.get_boundary_map(viewpoint_cam.image_name)
                densifier.update_cache(gaussians, viewpoint_cam, region_map, boundary_map, iteration)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="SG-3DGS Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # SG-3DGS specific arguments
    parser.add_argument("--sg3dgs_config", type=str, default=None,
                        help="Path to SG-3DGS YAML config file")
    parser.add_argument("--disable_sa_adc", action="store_true",
                        help="Disable SA-ADC (semantic-aware densification)")
    parser.add_argument("--disable_mfg_loss", action="store_true",
                        help="Disable MFG-Loss (feature-guided loss)")
    parser.add_argument("--disable_sp_reg", action="store_true",
                        help="Disable SP-Reg (structure-preserving regularization)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Load SG-3DGS config
    sg3dgs_config = load_sg3dgs_config(args.sg3dgs_config)

    # Apply command-line overrides
    if args.disable_sa_adc:
        sg3dgs_config["sa_adc"]["enabled"] = False
    if args.disable_mfg_loss:
        sg3dgs_config["mfg_loss"]["enabled"] = False
    if args.disable_sp_reg:
        sg3dgs_config["sp_reg"]["enabled"] = False

    print("=" * 60)
    print("SG-3DGS: Semantic-Guided 3D Gaussian Splatting")
    print("=" * 60)
    print(f"  SA-ADC:    {'ON' if sg3dgs_config['sa_adc']['enabled'] else 'OFF'}")
    print(f"  MFG-Loss:  {'ON' if sg3dgs_config['mfg_loss']['enabled'] else 'OFF'}")
    print(f"  SP-Reg:    {'ON' if sg3dgs_config['sp_reg']['enabled'] else 'OFF'}")
    print("=" * 60)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, sg3dgs_config)

    # All done
    print("\nSG-3DGS Training complete.")
