# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import dataclasses
import time
from pathlib import Path
from threading import Lock
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32

from ._renderer import Renderer, RenderTask
from .render_panel import RenderTabState, populate_general_render_tab


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Optional[Path] = None,
        mode: Literal["rendering", "training"] = "rendering",
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = "preparing"
        self.output_dir = output_dir if output_dir is not None else Path("./results")

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        self.render_modality = "rgb"

        # Initialize and populate GUIs.
        server.scene.set_global_visibility(True)
        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)
        server.gui.set_panel_label("basic viewer")
        server.gui.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )
        if self.mode == "training":
            self._init_training_tab()
            # self._populate_training_tab()

        self._init_rendering_tab()
        self._populate_rendering_tab()
        self.state = mode

    def _init_training_tab(self):
        self._training_tab_handles = {}
        self._training_folder = self.server.gui.add_folder("Training")

    def _populate_training_tab(self):
        server = self.server
        with self._training_folder:
            step_number = server.gui.add_number(
                "Step",
                min=0,
                max=1000000,
                step=1,
                disabled=True,
                initial_value=0,
            )
            pause_train_button = server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
                hint="Pause the training.",
            )
            resume_train_button = server.gui.add_button(
                "Resume",
                icon=viser.Icon.PLAYER_PLAY,
                visible=False,
                hint="Resume the training.",
            )

            @pause_train_button.on_click
            @resume_train_button.on_click
            def _(_) -> None:
                pause_train_button.visible = not pause_train_button.visible
                resume_train_button.visible = not resume_train_button.visible
                if self.state != "completed":
                    self.state = "paused" if self.state == "training" else "training"

            train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            train_util_slider.on_update(self.rerender)

        self._training_tab_handles = {
            "step_number": step_number,
            "pause_train_button": pause_train_button,
            "resume_train_button": resume_train_button,
            "train_util_slider": train_util_slider,
        }

    def _init_rendering_tab(self):
        # Allow subclasses to override for custom rendering table
        self.render_tab_state = RenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        # Allow subclasses to override for custom rendering table
        assert self.render_tab_state is not None, "Render tab state is not initialized"
        assert self._rendering_folder is not None, "Rendering folder is not initialized"

        with self._rendering_folder:

            with self.server.gui.add_folder(
                "4DGT Status", visible=True
            ) as self._stats_folder:
                self._stats_text = self.server.gui.add_markdown("")

            viewer_res_slider = self.server.gui.add_slider(
                "Viewer Res",
                min=256,
                max=512,
                step=32,
                initial_value=512,
                hint="Maximum resolution of the viewer rendered image.",
            )

            @viewer_res_slider.on_update
            def _(_) -> None:
                self.render_tab_state.viewer_res = int(viewer_res_slider.value)
                self.rerender(_)

            self._rendering_tab_handles["viewer_res_slider"] = viewer_res_slider

            time_slider = self.server.gui.add_slider( 
                "Time", 
                min = 0, 
                max = 1.0, 
                step = 0.067, 
                initial_value = 0.5, 
                hint = "The selected time within the input lifespan."
            )
            @time_slider.on_update 
            def _(_) -> None:
                self.render_tab_state.time_ratio = time_slider.value 
                self.rerender(_)

            frame_slider = self.server.gui.add_slider(
                "Frame",
                min=0,
                max=63,
                step=1,
                initial_value=0,
            )
            self.frame_slider = frame_slider
            
            @frame_slider.on_update
            def _(_) -> None:
                if hasattr(self, 'fourdgt_viewer'):
                    # Set camera to the selected index from input cameras
                    self.fourdgt_viewer.set_camera_to_input_index(int(frame_slider.value))
                    
                    # Update slider max if needed
                    num_cameras = self.fourdgt_viewer.get_input_camera_count()
                    if num_cameras > 0 and frame_slider.max != num_cameras - 1:
                        frame_slider.max = num_cameras - 1

            render_modality = self.server.gui.add_dropdown(
                "Modality",
                ("rgb", "depth", "normal", "motion"),
                initial_value="rgb",
            )

            @render_modality.on_update
            def _(event: viser.GuiEvent) -> None:
                self.render_modality = render_modality.value
                print(f"Render modality: {self.render_modality}")
                self.rerender(_)

            # Add scene navigation buttons if we have a fourdgt_viewer
            prev_button = self.server.gui.add_button(
                "Previous Scene", icon=viser.Icon.ARROW_AUTOFIT_LEFT
            )
            next_button = self.server.gui.add_button(
                "Next Scene", icon=viser.Icon.ARROW_AUTOFIT_RIGHT
            )
            
            @prev_button.on_click
            def _(_) -> None:
                if hasattr(self, 'fourdgt_viewer'):
                    self.fourdgt_viewer.switch_to_previous_scene()
                    self.rerender(_)
            
            @next_button.on_click
            def _(_) -> None:
                if hasattr(self, 'fourdgt_viewer'):
                    self.fourdgt_viewer.switch_to_next_scene()
                    self.rerender(_)

            recenter_button = self.server.gui.add_button(
                "Recenter", icon=viser.Icon.ARROW_AUTOFIT_RIGHT
            )
            @recenter_button.on_click
            def _(_) -> None:
                if hasattr(self, 'fourdgt_viewer'):
                    # Recenter the viewer to the original center camera
                    self.fourdgt_viewer.recenter_viewer()
                    self.rerender(_)


        # training tab handles should also be disabled during dumping video.
        extra_handles = self._rendering_tab_handles.copy()
        if self.mode == "training":
            extra_handles.update(self._training_tab_handles)

        # Disable the default general render tab
        # handles = populate_general_render_tab(
        #     self.server,
        #     output_dir=self.output_dir,
        #     folder=self._rendering_folder,
        #     render_tab_state=self.render_tab_state,
        #     extra_handles=extra_handles,
        # )
        # self._rendering_tab_handles.update(handles)

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.perf_counter()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        self._training_tab_handles["step_number"].value = step
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.perf_counter() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if (
            self.state == "training"
            and self._training_tab_handles["train_util_slider"].value != 1
        ):
            assert (
                self.render_tab_state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.render_tab_state.num_train_rays_per_sec
            view_s = self.render_tab_state.num_view_rays_per_sec
            train_util = self._training_tab_handles["train_util_slider"].value
            view_n = self.render_tab_state.viewer_res**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(
                        RenderTask("update", camera_state)
                    )

    def _after_render(self):
        # This function will be called each time render_fn is called.
        # It can be used to update the viewer panel.
        pass

    def complete(self):
        print("Training complete, disable training tab.")
        self.state = "completed"
        self._training_tab_handles["pause_train_button"].disabled = True
        self._training_tab_handles["resume_train_button"].disabled = True
        self._training_tab_handles["train_util_slider"].disabled = True
