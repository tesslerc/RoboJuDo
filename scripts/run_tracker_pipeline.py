"""Run a ProtoMotions tracker policy.

Extends the generic pipeline runner with tracker-specific features:
- CLI overrides for ONNX model, motion file, and motion index
- Default-pose mode: hold standing pose until R is pressed
- Auto-reset with configurable hold time after motion ends

Usage::

    python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \\
        --onnx-path /path/to/unified_pipeline.onnx \\
        --motion-path /path/to/motion.motion

    # Auto-reset after 5s hold:
    python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \\
        --onnx-path model.onnx --motion-path motion.motion \\
        --hold-seconds 5
"""

# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.utils.progress import ProgressBar

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a ProtoMotions tracker policy")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1_protomotions_tracker",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to the ONNX policy file (overrides config)",
    )
    parser.add_argument(
        "--motion-path",
        type=str,
        default=None,
        help="Path to the motion file (overrides config)",
    )
    parser.add_argument(
        "--motion-index",
        type=int,
        default=None,
        help="Index of motion clip within a multi-motion .pt library",
    )
    parser.add_argument(
        "--prepare-seconds",
        type=float,
        default=None,
        help="Duration of each prepare phase in seconds (default: 3s ramp + 5s blend)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help=(
            "Exit after motion ends + this many seconds of hold. "
            "Omit to run forever."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()

    # Override policy paths from CLI arguments.
    if args.onnx_path is not None and hasattr(cfg, "policy"):
        cfg.policy.onnx_path = args.onnx_path
    if args.motion_path is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_path = args.motion_path
    if args.motion_index is not None and hasattr(cfg, "policy"):
        cfg.policy.motion_index = args.motion_index

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare(prepare_seconds=args.prepare_seconds)
    elif pipeline._has_default_pose_mode:
        # In sim without prepare: hold default pose until R is pressed.
        pipeline._set_default_pose_mode(True)
        logger.warning("Sim mode — holding default pose, press R to start motion")

    # Compute max steps if --hold-seconds is set.
    max_steps = None
    motion_steps = None
    hold_steps = 0
    if args.hold_seconds is not None:
        # Get motion length from the policy's MotionPlayer (if available).
        inner_policy = getattr(pipeline.policy, "policy", pipeline.policy)
        player = getattr(inner_policy, "_player", None)
        if player is not None:
            motion_steps = player.total_frames
            hold_steps = int(args.hold_seconds * pipeline.freq)
            max_steps = motion_steps + hold_steps
            motion_secs = motion_steps / pipeline.freq
            logger.info(
                f"Will run {motion_steps} motion steps ({motion_secs:.1f}s) + "
                f"{hold_steps} hold steps ({args.hold_seconds:.1f}s) = "
                f"{max_steps} total"
            )
        else:
            logger.warning("--hold-seconds ignored: policy has no _player")

    step_count = 0
    hold_pbar = None
    while True:
        # Auto-reset: when hold time expires, simulate pressing R.
        # Skip if user manually triggered fade-out (wait for explicit fade-in).
        if max_steps is not None and step_count >= max_steps and not pipeline._user_fade_out:
            if hold_pbar is not None:
                hold_pbar.close()
                hold_pbar = None
            logger.info("Hold time reached — auto-resetting motion")
            # Trigger the same path as pressing R.
            pipeline._blend_out_active = False
            pipeline._blend_out_step = 0
            pipeline.policy.reset()
            pipeline._run_blend_in()
            step_count = 0
            continue

        # Start hold countdown when motion ends.
        if motion_steps is not None and step_count == motion_steps:
            hold_pbar = ProgressBar("Hold", hold_steps)
            logger.info("Motion ended — holding...")

        time_start = time.time()
        pipeline.step()
        step_count += 1

        # Reset step counter after any blend-in (manual or auto).
        if pipeline._blend_in_completed:
            pipeline._blend_in_completed = False
            step_count = 0
            if hold_pbar is not None:
                hold_pbar.close()
                hold_pbar = None

        if hold_pbar is not None:
            hold_pbar.update()
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
