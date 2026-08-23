"""Run continuous rendered TA-RWARE heuristic episodes for demo recording."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from typing import Any, cast

import gymnasium as gym
import tarware  # noqa: F401 - importing registers the TA-RWARE environments
from tarware.heuristic import heuristic_episode
from tarware.rendering import Viewer
from tarware.warehouse import RewardType, Warehouse

import pyglet
from pyglet.gl import gl

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
DEFAULT_MAX_STEPS = 500
HUD_HEIGHT = 92
HUD_BACKGROUND = (18, 22, 33)
HUD_PRIMARY = (255, 255, 255, 255)
HUD_SECONDARY = (150, 205, 255, 255)
HUD_ACCENT = (255, 200, 0, 255)
MINIMUM_CELL_PIXELS = 12


def relax_gdiplus_bitmap_argtypes() -> None:
    """Let pyglet 1.5.11 draw text on Windows with Python 3.12.

    `pyglet.font.win32` allocates a `c_byte` buffer but shares the GDI+ handle
    with `pyglet.image.codecs.gdiplus`, which declares that argument as
    `POINTER(c_ubyte)`. Python 3.12 ctypes rejects the mismatch. Widening the
    declaration to `c_void_p` accepts both buffer types.
    """
    if sys.platform != "win32":
        return
    from pyglet.image.codecs.gdiplus import gdiplus

    gdiplus.GdipCreateBitmapFromScan0.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]


class ShowcaseViewer(Viewer):
    """TA-RWARE viewer with larger cells and a heads-up display band."""

    def __init__(
        self, world_size: tuple[int, int], cell_pixels: int, fullscreen: bool = False
    ) -> None:
        super().__init__(world_size)
        self.grid_size = cell_pixels
        self.icon_size = max(8, cell_pixels * 2 // 3)
        self.world_height = 1 + self.rows * (self.grid_size + 1)
        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = self.world_height + HUD_HEIGHT
        self.window.set_size(self.width, self.height)
        self.window.set_caption("TA-RWARE showcase")
        self.window.on_close = self.request_stop
        if fullscreen:
            self.window.set_fullscreen(True)
            self.width = self.window.width
            self.height = self.window.height
        self.x_offset = max(
            0, (self.width - (1 + self.cols * (self.grid_size + 1))) // 2
        )
        self._labels = [
            pyglet.text.Label(
                "",
                font_name="Consolas",
                font_size=size,
                x=self.x_offset + 14,
                y=self.world_height + offset,
                anchor_x="left",
                anchor_y="center",
                color=color,
            )
            for size, offset, color in (
                (15, HUD_HEIGHT - 22, HUD_PRIMARY),
                (12, HUD_HEIGHT - 48, HUD_SECONDARY),
                (12, HUD_HEIGHT - 70, HUD_ACCENT),
            )
        ]

    def request_stop(self) -> None:
        """Ask the demo loop to finish instead of calling the upstream exit()."""
        self.isopen = False

    def render_frame(self, env: Warehouse, hud_lines: list[str]) -> bool:
        """Draw one warehouse frame plus the HUD band and swap buffers."""
        gl.glClearColor(1.0, 1.0, 1.0, 0.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        gl.glPushMatrix()
        gl.glTranslatef(float(self.x_offset), 0.0, 0.0)
        self._draw_grid()
        self._draw_goals(env)
        self._draw_shelfs(env)
        self._draw_agents(env)
        gl.glPopMatrix()
        self._draw_hud(hud_lines)
        self.window.flip()
        return self.isopen

    def _draw_hud(self, hud_lines: list[str]) -> None:
        gl.glColor3ub(255, 255, 255)
        top = self.world_height
        pyglet.graphics.draw(
            4,
            gl.GL_QUADS,
            (
                "v2f",
                (
                    0.0,
                    float(top),
                    float(self.width),
                    float(top),
                    float(self.width),
                    float(self.height),
                    0.0,
                    float(self.height),
                ),
            ),
            ("c4B", (*HUD_BACKGROUND, 255) * 4),
        )
        for label, text in zip(self._labels, hud_lines):
            label.text = text
            label.draw()


class ShowcaseWarehouse:
    """Proxy that adds live statistics, HUD rendering, and frame pacing."""

    def __init__(
        self,
        warehouse: Warehouse,
        environment_id: str,
        cell_pixels: int,
        frame_delay: float,
        fullscreen: bool = False,
    ) -> None:
        self._warehouse = warehouse
        self._environment_id = environment_id
        self._cell_pixels = cell_pixels
        self._frame_delay = frame_delay
        self._fullscreen = fullscreen
        self._viewer: ShowcaseViewer | None = None
        self.episode = 0
        self.seed = 0
        self.steps = 0
        self.deliveries = 0
        self.clashes = 0
        self.stucks = 0
        self.best_deliveries = 0
        self.best_episode = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._warehouse, name)

    def begin_episode(self, episode: int, seed: int) -> None:
        """Reset the per-episode counters shown in the HUD."""
        self.episode = episode
        self.seed = seed
        self.steps = 0
        self.deliveries = 0
        self.clashes = 0
        self.stucks = 0

    def finish_episode(self) -> None:
        """Record the running best after an episode ends."""
        if self.deliveries > self.best_deliveries:
            self.best_deliveries = self.deliveries
            self.best_episode = self.episode

    def step(self, action: Any) -> Any:
        result = self._warehouse.step(action)
        info = cast(dict[str, Any], result[4])
        self.steps += 1
        self.deliveries += int(info.get("shelf_deliveries", 0))
        self.clashes += int(info.get("clashes", 0))
        self.stucks += int(info.get("stucks", 0))
        return result

    def render(self, mode: str = "human") -> bool:
        if self._viewer is None:
            world_size = cast(tuple[int, int], self._warehouse.grid_size)
            self._viewer = ShowcaseViewer(
                world_size, self._cell_pixels, self._fullscreen
            )
            self._warehouse.renderer = self._viewer
        is_open = self._viewer.render_frame(self._warehouse, self._hud_lines())
        if not is_open:
            raise KeyboardInterrupt
        if self._frame_delay > 0.0:
            time.sleep(self._frame_delay)
        return is_open

    def _hud_lines(self) -> list[str]:
        busy = sum(1 for agent in self._warehouse.agents if agent.busy)
        best = (
            f"best {self.best_deliveries} (ep {self.best_episode})"
            if self.best_episode
            else "best -"
        )
        return [
            f"EPISODE {self.episode}   seed {self.seed}   step {self.steps}",
            f"deliveries {self.deliveries}   clashes {self.clashes}   "
            f"stucks {self.stucks}   busy {busy}/{self._warehouse.num_agents}",
            f"{self._environment_id}   {best}",
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id",
        default=ENVIRONMENT_ID,
        help="Registered TA-RWARE id, e.g. tarware-small-6agvs-3pickers-partialobs-v1.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Steps per episode. TA-RWARE registers every layout with 500.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run. 0 repeats forever until the window closes.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed of the first episode."
    )
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Replay the same seed every episode instead of incrementing it.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frame pacing for recording. Use 0 to run as fast as possible.",
    )
    parser.add_argument(
        "--cell-pixels",
        type=int,
        help="Warehouse cell size in pixels. Defaults to fitting the target window.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fill the whole screen and ignore the window size options.",
    )
    parser.add_argument(
        "--shelf-rows",
        type=int,
        help="Build a recording-only layout with this many shelf block rows.",
    )
    parser.add_argument(
        "--shelf-columns",
        type=int,
        help="Shelf block columns for the recording-only layout. Must be odd.",
    )
    parser.add_argument("--column-height", type=int, default=8)
    parser.add_argument("--agvs", type=int, default=10)
    parser.add_argument("--pickers", type=int, default=5)
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--window-width", type=int, default=1440)
    parser.add_argument("--window-height", type=int, default=780)
    return parser.parse_args()


def register_demo_layout(
    shelf_rows: int,
    shelf_columns: int,
    column_height: int,
    num_agvs: int,
    num_pickers: int,
    max_steps: int,
    queue_size: int,
) -> str:
    """Register a recording-only warehouse shape and return its environment id.

    TA-RWARE only registers five portrait layouts. A wide, short warehouse fills
    a 16:9 screen far better. These shapes are outside the pinned experiment set
    and must not be used to produce comparable results.
    """
    if shelf_columns % 2 == 0:
        raise SystemExit("--shelf-columns must be odd; TA-RWARE asserts this.")
    environment_id = (
        f"tarware-demo-{shelf_rows}x{shelf_columns}x{column_height}"
        f"-{num_agvs}agvs-{num_pickers}pickers-v1"
    )
    gym.register(
        id=environment_id,
        entry_point="tarware.warehouse:Warehouse",
        kwargs={
            "column_height": column_height,
            "shelf_rows": shelf_rows,
            "shelf_columns": shelf_columns,
            "num_agvs": num_agvs,
            "num_pickers": num_pickers,
            "request_queue_size": queue_size,
            "max_inactivity_steps": None,
            "max_steps": max_steps,
            "reward_type": RewardType.INDIVIDUAL,
            "observation_type": "partial",
        },
    )
    return environment_id


def screen_size() -> tuple[int, int]:
    """Return the pixel size of the default screen."""
    screen = pyglet.canvas.get_display().get_default_screen()
    return int(screen.width), int(screen.height)


def fitted_cell_pixels(world_size: tuple[int, int], width: int, height: int) -> int:
    """Return the largest cell size whose window fits the requested box."""
    rows, cols = world_size
    by_width = (width - 1) // cols - 1
    by_height = (height - HUD_HEIGHT - 1) // rows - 1
    return max(MINIMUM_CELL_PIXELS, min(by_width, by_height))


def main() -> None:
    args = parse_args()
    relax_gdiplus_bitmap_argtypes()
    if args.shelf_rows is not None or args.shelf_columns is not None:
        environment_id = register_demo_layout(
            args.shelf_rows if args.shelf_rows is not None else 1,
            args.shelf_columns if args.shelf_columns is not None else 7,
            args.column_height,
            args.agvs,
            args.pickers,
            args.max_steps,
            args.queue_size,
        )
    else:
        environment_id = args.env_id
    env = gym.make(
        environment_id,
        disable_env_checker=True,
        max_steps=args.max_steps,
    )
    warehouse = cast(Warehouse, env.unwrapped)
    if args.fullscreen:
        target_width, target_height = screen_size()
    else:
        target_width, target_height = args.window_width, args.window_height
    cell_pixels = args.cell_pixels or fitted_cell_pixels(
        cast(tuple[int, int], warehouse.grid_size),
        target_width,
        target_height,
    )
    frame_delay = 1.0 / args.fps if args.fps > 0.0 else 0.0
    showcase = ShowcaseWarehouse(
        warehouse, environment_id, cell_pixels, frame_delay, args.fullscreen
    )

    print(f"{environment_id} | {args.max_steps} steps/episode | {cell_pixels}px cells")
    print("Close the viewer window or press Ctrl+C to stop.")
    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            episode += 1
            seed = args.seed if args.same_seed else args.seed + episode - 1
            showcase.begin_episode(episode, seed)
            _, total_return, _ = heuristic_episode(showcase, render=True, seed=seed)
            showcase.finish_episode()
            print(
                f"episode {episode:>3} | seed {seed:>3} | "
                f"steps {showcase.steps:>4} | deliveries {showcase.deliveries:>3} | "
                f"clashes {showcase.clashes:>3} | stucks {showcase.stucks:>3} | "
                f"return {float(total_return):.1f}"
            )
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
