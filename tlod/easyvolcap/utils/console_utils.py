# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import socket


# fmt: off
class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"

essential_packages = {
    'tqdm': 'tqdm',
    'ujson': 'ujson',
    'IPython': 'IPython',
    'ruamel.yaml': 'ruamel.yaml',
    'yapf': 'yapf',
    'h5py': 'h5py',
    'psutil': 'psutil',
    'PyYAML': 'yaml',
    'addict': 'addict',
    'websockets': 'websockets',
}

try:
    for _, import_name in essential_packages.items():
        __import__(import_name)
except ImportError as e:
    print(f'{Colors.YELLOW}{Colors.BOLD}Missing package: {Colors.RED}{Colors.BOLD}{e}{Colors.YELLOW}{Colors.BOLD}, trying to hot install using pip...{Colors.END}')
    import subprocess
    import sys
    subprocess.call([sys.executable, '-m', 'ensurepip'])
    subprocess.call([sys.executable, '-m', 'pip', 'install', *essential_packages.keys()])


import argparse
import ast
import atexit
import os
import re
import shutil
import sys
import time
import warnings
from bdb import BdbQuit
from collections import deque
from functools import partial
from io import BytesIO, StringIO
from os.path import (  # noqa: F401
    abspath,
    basename,  # noqa: F401
    dirname,
    exists,
    getsize,  # noqa: F401
    isdir,  # noqa: F401
    isfile,  # noqa: F401
    join,
    relpath,  # noqa: F401
    split,  # noqa: F401
    splitext,  # noqa: F401
)
from typing import (  # noqa: F401
    Any,  # noqa: F401
    Callable,
    cast,  # noqa: F401
    Dict,  # noqa: F401
    IO,
    Iterable,  # noqa: F401
    List,
    Mapping,
    Optional,
    Sequence,  # noqa: F401
    Type,  # noqa: F401
    Union,
)

import numpy as np
import ujson as json

from rich import pretty, traceback
from rich.console import Console
from rich.live import Live
from rich.pretty import pretty_repr
from rich.progress import (
    BarColumn,
    filesize,
    Progress,
    ProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Column, Table
from rich.text import Text
from ruamel.yaml import YAML
from tqdm.rich import FractionColumn
from tqdm.std import tqdm as std_tqdm

from .base_utils import default_dotdict, dotdict

host_name = socket.gethostname()


try:
    import pdbr
    from pdbr import RichPdb
    pdbr_theme = 'ansi_dark'
    pdbr.utils.set_traceback(pdbr_theme)
    RichPdb._theme = pdbr_theme
except ImportError:
    pass
# fmt: on


class MyYAML(YAML):
    def dumps(self, obj: Union[dict, dotdict]):
        if isinstance(obj, dotdict):
            obj = obj.to_dict()
        buf = BytesIO()
        self.dump(obj, buf)  # ?: is the dumping also in utf-8?
        return buf.getvalue().decode(encoding="utf-8", errors="strict")[
            :-1
        ]  # remove \n


yaml = MyYAML()
yaml.default_flow_style = os.environ.get("EASYVOLCAP_DEFAULT_FLOW_STYLE", None)

warnings.filterwarnings("ignore")
os.environ["PYTHONPATH"] = (
    os.environ.get("PYTHONPATH", "") + ":" + join(dirname(__file__), "..", "..", "..")
)
os.environ["PYTHONBREAKPOINT"] = "tlod.easyvolcap.utils.console_utils.set_trace"

auto_refresh = (
    os.environ["EASYVOLCAP_PROGRESS_AUTO_REFRESH"].lower() == "true"
    if "EASYVOLCAP_PROGRESS_AUTO_REFRESH" in os.environ
    else True
)
force_terminal = (
    os.environ["EASYVOLCAP_CONSOLE_FORCE_TERMINAL"].lower() == "true"
    if "EASYVOLCAP_CONSOLE_FORCE_TERMINAL" in os.environ
    else None
)
slim_width = (
    int(os.environ["EASYVOLCAP_CONSOLE_SLIM_WIDTH"])
    if "EASYVOLCAP_CONSOLE_SLIM_WIDTH" in os.environ
    else None
)
verbose_width = (
    int(os.environ["EASYVOLCAP_CONSOLE_VERBOSE_WIDTH"])
    if "EASYVOLCAP_CONSOLE_VERBOSE_WIDTH" in os.environ
    else None
)
slim_log_time = (
    bool(os.environ["EASYVOLCAP_CONSOLE_SLIM_LOG_TIME"])
    if "EASYVOLCAP_CONSOLE_SLIM_LOG_TIME" in os.environ
    else True
)
slim_log_path = (
    bool(os.environ["EASYVOLCAP_CONSOLE_SLIM_LOG_PATH"])
    if "EASYVOLCAP_CONSOLE_SLIM_LOG_PATH" in os.environ
    else True
)
slim_time_format = "%H:%M:%S"
# slim_time_format = ''
verbose_time_format = "%Y-%m-%d %H:%M:%S.%f"
do_nothing_console = Console(file=StringIO(), stderr=StringIO())
console = Console(
    soft_wrap=True,
    tab_size=4,
    log_time_format=slim_time_format,
    width=slim_width,
    log_time=slim_log_time,
    log_path=slim_log_path,
    force_terminal=force_terminal,
)
progress = Progress(
    console=console, expand=True, auto_refresh=auto_refresh
)
live = Live(console=console, refresh_per_second=2)
traceback.install(
    console=console, width=slim_width, indent_guides=False
)
pretty.install(console=console)

NoneType = type(None)

# PYTORCH_CUDA_ALLOC_CONF = os.environ['PYTORCH_CUDA_ALLOC_CONF'].split(',') if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ else []
# PYTORCH_CUDA_ALLOC_CONF = ','.join(PYTORCH_CUDA_ALLOC_CONF + ['expandable_segments:True'])
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF

# fmt: off
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    String,
    Text as TextToken,
    Token,
)

from rich import pretty
from rich._loop import loop_last
from rich.console import (  # noqa: F401
    Console,
    ConsoleOptions,
    ConsoleRenderable,
    RenderResult,
)
from rich.constrain import Constrain
from rich.highlighter import ReprHighlighter
from rich.panel import Panel
from rich.theme import Theme


def traceback_panel_rich_console(
    self: traceback.Traceback, console: Console, options: ConsoleOptions
) -> RenderResult:
    theme = self.theme
    background_style = theme.get_background_style()
    token_style = theme.get_style_for_token

    traceback_theme = Theme(
        {
            "pretty": token_style(TextToken),
            "pygments.text": token_style(Token),
            "pygments.string": token_style(String),
            "pygments.function": token_style(Name.Function),
            "pygments.number": token_style(Number),
            "repr.indent": token_style(Comment) + Style(dim=True),
            "repr.str": token_style(String),
            "repr.brace": token_style(TextToken) + Style(bold=True),
            "repr.number": token_style(Number),
            "repr.bool_true": token_style(Keyword.Constant),
            "repr.bool_false": token_style(Keyword.Constant),
            "repr.none": token_style(Keyword.Constant),
            "scope.border": token_style(String.Delimiter),
            "scope.equals": token_style(Operator),
            "scope.key": token_style(Name),
            "scope.key.special": token_style(Name.Constant) + Style(dim=True),
        },
        inherit=False,
    )

    highlighter = ReprHighlighter()
    for last, stack in loop_last(reversed(self.trace.stacks)):
        if stack.frames:
            stack_renderable: ConsoleRenderable = self._render_stack(stack)
            # stack_renderable: ConsoleRenderable = Panel(
            #     self._render_stack(stack),
            #     title="[traceback.title]Traceback [dim](most recent call last)",
            #     style=background_style,
            #     border_style="traceback.border",
            #     expand=True,
            #     padding=(0, 1),
            # )
            stack_renderable = Constrain(stack_renderable, self.width)
            with console.use_theme(traceback_theme):
                yield stack_renderable
        if stack.syntax_error is not None:
            with console.use_theme(traceback_theme):
                yield Constrain(
                    Panel(
                        self._render_syntax_error(stack.syntax_error),
                        style=background_style,
                        border_style="traceback.border.syntax_error",
                        expand=True,
                        padding=(0, 1),
                        width=self.width,
                    ),
                    self.width,
                )
            yield Text.assemble(
                (f"{stack.exc_type}: ", "traceback.exc_type"),
                highlighter(stack.syntax_error.msg),
            )
        elif stack.exc_value:
            yield Text.assemble(
                (f"{stack.exc_type}: ", "traceback.exc_type"),
                highlighter(stack.exc_value),
            )
        else:
            yield Text.assemble((f"{stack.exc_type}", "traceback.exc_type"))

        if not last:
            if stack.is_cause:
                yield Text.from_markup(
                    "\n[i]The above exception was the direct cause of the following exception:\n",
                )
            else:
                yield Text.from_markup(
                    "\n[i]During handling of the above exception, another exception occurred:\n",
                )

traceback.Traceback.__rich_console__ = traceback_panel_rich_console # monkey patch the console
# fmt: on


class WithoutLive:
    def __enter__(self):
        stop_live()
        stop_prog()

    def __exit__(self, exc_type, exc_val, exc_tb):
        start_live()
        start_prog()


def stop_live():
    global live
    if live is None:
        return
    live.stop()
    live = None


def start_live():
    global live
    if live is not None:
        live.start()
        return
    live = Live(console=console, refresh_per_second=1)
    live.start()


def stop_prog():
    global progress
    if progress is None:
        return
    progress.stop()
    progress = None


def start_prog():
    global progress
    if progress is not None:
        return
    progress = Progress(console=console, expand=True)


def stacktrace(extra_lines=0, **kwargs):
    # Print colorful stacktrace
    kwargs.update(dict(extra_lines=extra_lines))  # noqa: C408

    console.print(
        traceback.Traceback(
            width=slim_width,
            extra_lines=3,
            theme=None,
            word_wrap=False,
            show_locals=False,
            suppress=(),
            max_frames=100,
            indent_guides=False,
        )
    )


breakpoint_disabled = False


def disable_breakpoint():
    global breakpoint_disabled
    breakpoint_disabled = True


def enable_breakpoint():
    global breakpoint_disabled
    breakpoint_disabled = False


progress_disabled = False
standard_console = console


def disable_console():
    global console, standard_console, do_nothing_console, progress, live
    console = do_nothing_console
    live.console = do_nothing_console
    progress.live.console = do_nothing_console


def enable_console():
    global console, standard_console, do_nothing_console, progress, live
    console = standard_console
    live.console = standard_console
    progress.live.console = standard_console


def disable_progress():
    global progress_disabled
    progress_disabled = True


def enable_progress():
    global progress_disabled
    progress_disabled = False


verbose_log = False


def disable_verbose_log():
    global verbose_log
    verbose_log = False
    console.width = slim_width
    console._log_render.show_time = slim_log_time
    console._log_render.show_path = slim_log_path
    console._log_render.time_format = slim_time_format


def enable_verbose_log():
    global verbose_log
    verbose_log = True
    console.width = verbose_width
    console._log_render.show_time = True
    console._log_render.show_path = True
    console._log_render.time_format = verbose_time_format


disable_verbose_log()


def set_trace(*args, **kwargs):
    if breakpoint_disabled:
        return
    stop_live()
    stop_prog()

    if "RichPdb" in globals():
        rich_pdb = RichPdb()
        rich_pdb.set_trace(sys._getframe(1))
        # else:
        #     ipdb.set_trace(sys._getframe(1))


def post_mortem(*args, **kwargs):
    stop_live()
    stop_prog()
    if "pdbr" in globals():
        pdbr.post_mortem()
    # else:
    #     ipdb.post_mortem()  # break on the last exception's stack for inpection


def line(obj):
    """
    Represent objects in oneline for prettier printing
    """
    s = pretty_repr(obj, indent_size=0)
    s = s.replace("\n", " ")
    s = re.sub(" {2,}", " ", s)
    return s


def path(string):  # add path markup
    string = str(string)
    if exists(string):
        return Text(
            string,
            style=Style(bold=True, color="blue", link=f"file://{abspath(string)}"),
        )
    else:
        # return Text(string, style=Style(bold=True, color='blue'))
        return blue(string)


def red(string: str) -> str:
    return f"[red bold]{string}[/]"


def blue(string: str) -> str:
    return f"[blue bold]{string}[/]"


def cyan(string: str) -> str:
    return f"[cyan bold]{string}[/]"


def pink(string: str) -> str:
    return f"[bright_magenta bold]{string}[/]"


def green(string: str) -> str:
    return f"[green bold]{string}[/]"


def yellow(string: str) -> str:
    return f"[yellow bold]{string}[/]"


def magenta(string: str) -> str:
    return f"[magenta bold]{string}[/]"


def color(string: str, color: str):
    return f"[{color} bold]{string}[/]"


def bold(string: str):
    return f"[bold]{string}[/]"


def red_slim(string: str) -> str:
    return f"[red]{string}[/]"


def blue_slim(string: str) -> str:
    return f"[blue]{string}[/]"


def cyan_slim(string: str) -> str:
    return f"[cyan]{string}[/]"


def pink_slim(string: str) -> str:
    return f"[bright_magenta]{string}[/]"


def green_slim(string: str) -> str:
    return f"[green]{string}[/]"


def yellow_slim(string: str) -> str:
    return f"[yellow]{string}[/]"


def magenta_slim(string: str) -> str:
    return f"[magenta]{string}[/]"


def color_slim(string: str, color: str):
    return f"[{color}]{string}[/]"


def slim(string: str):
    return f"{string}"


def markup_to_ansi(string: str, end="") -> str:
    """
    Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with console.capture() as out:
        console.print(string, soft_wrap=True, end=end)
    return out.get()


def get_log_prefix(
    back=2,
    module_color=blue,
    func_color=green,
):
    frame = sys._getframe(back)  # with another offset
    func = frame.f_code.co_name
    module = frame.f_globals["__name__"] if frame is not None else ""
    return module_color(module) + " -> " + func_color(func) + ":"


def log(
    *stuff,
    back=1,
    file: Optional[IO[str]] = None,
    no_prefix=False,
    module_color=blue,
    func_color=green,
    console: Optional[Console] = console,
    **kwargs,
):
    """
    Perform logging using the built in shared logger
    """

    writer = (
        console
        if file is None
        else Console(
            file=file, soft_wrap=True, tab_size=4, log_time_format=verbose_time_format
        )
    )  # shared
    writer._log_render.time_format = (
        verbose_time_format if verbose_log else slim_time_format
    )
    if no_prefix or not verbose_log:
        writer.log(*stuff, _stack_offset=2, **kwargs)
    else:
        writer.log(
            get_log_prefix(back + 1, module_color, func_color),
            *stuff,
            _stack_offset=2,
            **kwargs,
        )


def run(
    cmd,
    quite=False,
    dry_run=False,
    skip_failed=False,
    invocation=os.system,  # or subprocess.run
):
    """
    Run a shell command and print the command to the console.

    Args:
        cmd (str or list): The command to run. If a list, it will be joined with spaces.
        quite (bool): If True, suppress console output.
        dry_run (bool): If True, print the command but do not execute it.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.

    Returns:
        None
    """
    if isinstance(cmd, list):
        cmd = " ".join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = "cyan" if not cmd.startswith("rm") else "red"
        cmd_color = "green" if dry_run else cmd_color
        dry_msg = magenta("[dry_run]: ") if dry_run else ""
        log(
            yellow(func),
            "->",
            green(invocation.__name__) + ":",
            dry_msg + color(cmd, cmd_color),
            no_prefix=True,
        )
        # print(color(cmd, cmd_color), soft_wrap=False)
    if not dry_run:
        code = invocation(cmd)
    else:
        code = 0
    if code != 0 and not skip_failed:
        log(red(code), "<-", yellow(func) + ":", red(cmd), no_prefix=True)
        # print(red(cmd), soft_wrap=True)
        raise RuntimeError(f"{code} <- {func}: {cmd}")
    else:
        return code  # or output


def read(cmd: str, *args, **kwargs):
    def get_output(cmd: str):
        # return subprocess.run(cmd.split(' '), stdout=subprocess.PIPE, shell=True).stdout
        return os.popen(cmd).read()

    kwargs["skip_failed"] = True
    return run(cmd, *args, invocation=get_output, **kwargs)


def run_if_not_exists(cmd, outname, *args, **kwargs):
    # whether a file exists, whether a directory has more than 3 elements
    # if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
    if os.path.exists(outname):
        log(yellow("Skipping:"), cyan(cmd))
    else:
        run(cmd, *args, **kwargs)


def catch_throw(fatal: Union[Callable, bool] = True):
    def wrapper(func: Callable):
        # This function catches errors and stops the execution for easier inspection
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, BdbQuit):
                    return  # so that nested catch_throw will respect each other
                log(red(f"Runtime exception: {e}"))
                stacktrace()
                post_mortem()
                if fatal:
                    exit(1)  # catched variable

        return inner

    if callable(fatal):
        func = fatal
        fatal = True
        return wrapper(func)  # just a regular function decorator
    else:
        fatal = fatal  # whether this is a fatal post_mortem
        return wrapper


def print(
    *stuff,
    sep: str = " ",
    end: str = "\n",
    file: Optional[IO[str]] = None,
    flush: bool = False,
    console: Optional[Console] = console,
    **kwargs,
):
    r"""
    Print object(s) supplied via positional arguments.
    This function has an identical signature to the built-in print.
    For more advanced features, see the :class:`~rich.console.Console` class.

    Args:
        sep (str, optional): Separator between printed objects. Defaults to " ".
        end (str, optional): Character to write at end of output. Defaults to "\\n".
        file (IO[str], optional): File to write to, or None for stdout. Defaults to None.
        flush (bool, optional): Has no effect as Rich always flushes output. Defaults to False.

    """
    writer = (
        console
        if file is None
        else Console(
            file=file, soft_wrap=True, tab_size=4, log_time_format=verbose_time_format
        )
    )  # shared
    writer.print(*stuff, sep=sep, end=end, **kwargs)


class ColoredLogger:
    # ColoredLogger.is_main_process = is_main_process

    def __init__(self, prefix: str = "", postfix: str = ""):
        self.prefix = prefix
        self.postfix = postfix

    def info(self, *things: str, **kwargs):
        from .dist_utils import is_main_process

        if not is_main_process():
            return
        log(self.prefix + str(" ".join([str(t) for t in things])) + self.postfix)

    def debug(self, *things: str, **kwargs):
        from .dist_utils import is_main_process

        if not is_main_process():
            return
        log(self.prefix + str(" ".join([str(t) for t in things])) + self.postfix)

    def warn(self, *things: str, **kwargs):
        from .dist_utils import is_main_process

        if not is_main_process():
            return
        log(
            yellow(self.prefix + str(" ".join([str(t) for t in things])) + self.postfix)
        )

    def error(self, *things: str, **kwargs):
        from .dist_utils import is_main_process

        if not is_main_process():
            return
        log(red(self.prefix + str(" ".join([str(t) for t in things])) + self.postfix))


def cached_log_stream(filename: str, mode="w", encoding="utf-8"):
    from ...misc.io_helper import pathmgr

    pathmgr.mkdirs(dirname(filename))
    fp = pathmgr.open(filename, mode, encoding=encoding)
    atexit.register(fp.close)
    return fp


class DualOutput:
    def __init__(self, file_path, mode="w", encoding="utf-8"):
        # Use the original stdout
        # self.stdout = sys.stdout
        self.stdout = sys.stdout
        self.encoding = encoding
        self.mode = mode
        # Open the file
        if file_path.startswith("manifold://"):
            self.file = file_path
            self.local_file = file_path.replace("manifold://", "/tmp/")
            print(f"Registering filename: {self.file}")
            self.local_file_pointer = cached_log_stream(self.local_file, mode, encoding)
        else:
            self.file = cached_log_stream(file_path, mode, encoding=encoding)

    def write(self, message):
        from ...misc.io_helper import pathmgr

        # Write to stdout
        # self.stdout.write(message)
        self.stdout.write(message)
        if isinstance(self.file, str):
            # Write to the file
            # with pathmgr.opena(self.file, "w", encoding=self.encoding) as f:
            #     f.write(message)
            self.local_file_pointer.write(message)
            self.local_file_pointer.flush()
            pathmgr.copy_from_local(self.local_file, self.file, overwrite=True)
        else:
            self.file.write(message)

    def flush(self):
        # Flush both the file and stdout
        self.stdout.flush
        # self.stdout.flush()
        if not isinstance(self.file, str):
            self.file.flush()

    def close(self):
        # Close the file
        if not isinstance(self.file, str):
            self.file.close()


class logging:
    mapping: Mapping[str, ColoredLogger] = {}

    @staticmethod
    def get_logger(name: str):
        if name not in logging.mapping:
            logging.mapping[name] = ColoredLogger(
                # prefix=f"[{name}] " if len(name) else ""
            )
        return logging.mapping[name]

    @staticmethod
    def setup_logging(save_path: str, *args, **kwargs):
        global console
        from .dist_utils import is_main_process

        if is_main_process():
            # os.makedirs(dirname(save_path), exist_ok=True)
            console.file = DualOutput(save_path)
            log(f"Logs will be backed up into: {blue(save_path)}")


logger = logging.get_logger("")

# https://github.com/tqdm/tqdm/blob/master/tqdm/rich.py
# this is really nice
# if we want to integrate this into our own system, just import the tqdm from here


class PathColumn(ProgressColumn):
    def __init__(self, **kwargs):
        filename, line_no, locals = console._caller_frame_info(2)
        link_path = None if filename.startswith("<") else os.path.abspath(filename)
        path = filename.rpartition(os.sep)[-1]
        path_text = Text(style="log.path")
        path_text.append(path, style=f"link file://{link_path}" if link_path else "")
        if line_no:
            path_text.append(":")
            path_text.append(
                f"{line_no}",
                style=f"link file://{link_path}#{line_no}" if link_path else "",
            )
        self.path_text = path_text
        super().__init__(**kwargs)

    def render(self, task):
        return self.path_text


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_scale=False, unit_divisor=1000, **kwargs):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__(**kwargs)

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"  ?  {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        # precision = 3 if unit == 1 else 6
        ratio = speed / unit

        precision = 3 - int(np.log(ratio) / np.log(10))
        precision = max(0, precision)
        return Text(
            f"{ratio:,.{precision}f} {suffix}{self.unit}/s", style="progress.data.speed"
        )


class PrefixColumn(ProgressColumn):
    def __init__(self, content: str = None, **kwargs):
        self.content = content
        super().__init__(**kwargs)

    def render(self, task):
        if self.content is not None:
            return self.content
        else:
            log_prefix = get_log_prefix(back=3)
            return log_prefix


class TimeColumn(ProgressColumn):
    def render(self, task):
        log_time = console.get_datetime()
        log_time_display = Text(
            log_time.strftime(verbose_time_format if verbose_log else slim_time_format),
            style="log.time",
        )
        return log_time_display


class tqdm_rich(std_tqdm):
    def __init__(self, *args, back=2, **kwargs):
        # This popping happens before initiating tqdm
        _prog = kwargs.pop("progress", None)

        # Thanks! tqdm!
        super().__init__(*args, **kwargs)
        self.disable = self.disable or progress_disabled
        if self.disable:
            return

        # Whatever for now
        stop_live()
        start_prog()
        _prog = _prog if _prog is not None else progress

        # Use the predefined progress object
        d = self.format_dict
        self._prog = _prog
        self._prog.columns = (
            ((TimeColumn(),) if verbose_log or slim_log_time else ())
            + (
                (PrefixColumn(content=get_log_prefix(back=back)),)
                if verbose_log
                else ()
            )
            + (
                "[progress.description]{task.description}"
                "[progress.percentage]{task.percentage:>4.0f}%",
                BarColumn(),
                FractionColumn(
                    unit_scale=d["unit_scale"], unit_divisor=d["unit_divisor"]
                ),
                TimeElapsedColumn(),
                "<",
                TimeRemainingColumn(),
                RateColumn(
                    unit=d["unit"],
                    unit_scale=d["unit_scale"],
                    unit_divisor=d["unit_divisor"],
                ),
            )
            + (
                (PathColumn(table_column=Column(ratio=1.0, justify="right")),)
                if verbose_log or slim_log_path
                else ()
            )
        )
        self._prog.start()
        self._task_id = self._prog.add_task(self.desc or "", **d)

    def close(self):
        if self.disable:
            return
        self.disable = True  # prevent multiple closures

        self.display(refresh=True)
        if self._prog.finished:
            self._prog.stop()
            for task_id in self._prog.task_ids:
                self._prog.remove_task(task_id)

            # Whatever for now
            stop_prog()

    def clear(self, *_, **__):
        pass

    def display(self, refresh=True, *_, **__):
        if not hasattr(self, "_prog"):
            return
        if self._task_id not in self._prog.task_ids:
            return
        self._prog.update(
            self._task_id, completed=self.n, description=self.desc, refresh=refresh
        )

    def reset(self, total=None):
        """
        Resets to 0 iterations for repeated use.

        Parameters
        ----------
        total  : int or float, optional. Total to use for the new bar.
        """
        if hasattr(self, "_prog"):
            self._prog.reset(self._task_id, total=total)
        super(tqdm_rich, self).reset(total=total)


use_std_tqdm = os.environ.get("EASYVOLCAP_USE_STD_TQDM", "").lower() == "true"
if use_std_tqdm:
    tqdm = std_tqdm
else:
    tqdm = tqdm_rich


def time_function(sync_cuda: bool = True):
    """Decorator: time a function call"""

    def inner(*args, func: Callable = lambda x: x, **kwargs):
        start = time.perf_counter()
        ret = func(*args, **kwargs)
        if sync_cuda:
            import torch  # don't want to place this outside

            torch.cuda.synchronize()
        end = time.perf_counter()
        name = getattr(func, "__name__", repr(func))
        log(name, f"{(end - start) * 1000:8.3f} ms", back=2)
        return ret

    def wrapper(func: Callable):
        return partial(inner, func=func)

    if isinstance(sync_cuda, bool):
        return wrapper
    else:
        func = sync_cuda
        sync_cuda = True
        return partial(inner, func=func)


class Timer:
    def __init__(
        self,
        name="base",
        exp_name="",
        record_dir: str = "data/timing",
        disabled: bool = False,
        sync_cuda: bool = True,
        record_to_file: bool = False,
    ):
        self.sync_cuda = sync_cuda
        self.disabled = disabled
        self.name = name
        self.exp_name = exp_name
        self.start_time = time.perf_counter()
        self.start()

        self.record_to_file = record_to_file
        if self.record_to_file:
            self.timing_record = dotdict()

        self.event_acc = dotdict()
        self.event_last = dotdict()
        self.event_denom = dotdict()

    def __enter__(self):
        self.start()

    def start(self):
        if self.disabled:
            return self
        if self.sync_cuda:
            import torch  # don't want to place this outside

            try:
                torch.cuda.synchronize()
            except:  # noqa: B001
                pass
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.stop()

    def stop(self, print=True, back=2):
        if self.disabled:
            return 0
        if self.sync_cuda:
            import torch  # don't want to place this outside

            try:
                torch.cuda.synchronize()
            except:  # noqa: B001
                pass
        start = self.start_time
        end = time.perf_counter()
        if print:
            log(
                f"{(end - start) * 1000:8.3f} ms", self.name, back=back
            )
        return end - start

    def record(self, event: str = "", log_interval: float = -1):
        if self.disabled:
            return 0
        self.name = event

        diff = self.stop(print=bool(event) and log_interval <= 0, back=3)
        curr = time.perf_counter()
        acc = self.event_acc.get(event, 0)
        last = self.event_last.get(event, 0)
        denom = self.event_denom.get(event, 0)

        if (
            (curr - last) > log_interval and log_interval > 0
        ):
            log(f"{(acc + diff) / (denom + 1) * 1000:8.3f} ms", event, back=3)
            self.event_acc[event] = 0
            self.event_denom[event] = 0
            self.event_last[event] = curr
        else:
            self.event_acc[event] = acc + diff
            self.event_denom[event] = denom + 1

        if self.record_to_file and event:
            if event not in self.timing_record:
                self.timing_record[event] = []
            self.timing_record[event].append(diff)

            with open(join(self.record_dir, f"{self.exp_name}.json"), "w") as f:
                json.dump(self.timing_record, f, indent=4)

        self.start()
        return diff

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


rows = None


def display_table(
    states: dotdict,
    styles=default_dotdict(  # noqa: B008
        NoneType,
        {
            "eta": "cyan",
            "epoch": "cyan",
            "img_loss": "magenta",
            "psnr": "magenta",
            "loss": "magenta",
            "data": "blue",
            "batch": "blue",
            "g_ratio": "green",
        },
    ),
    maxlen=5,
):
    def create_table(
        columns: List[str],
        rows: List[List[str]] = (),
        styles=default_dotdict(NoneType),  # noqa: B008
    ):
        try:
            from easyvolcap.engine import cfg

            title = cfg.exp_name
        except Exception:
            title = None
        table = Table(
            title=title, show_footer=True, show_header=False, box=None
        )  # move the row names down at the bottom
        for col in columns:
            table.add_column(
                footer=Text(col, styles[col]), style=styles[col], justify="center"
            )
        for row in rows:
            table.add_row(*row)
        return table

    keys = list(states.keys())
    values = list(map(str, states.values()))
    width, height = shutil.get_terminal_size(fallback=(120, 50))
    maxlen = max(min(height - 8, maxlen), 1)  # 5 would fill the terminal

    global rows
    if rows is None:
        rows = deque(maxlen=maxlen)
    if rows.maxlen != maxlen:
        rows = deque(
            list(rows)[-maxlen + 1 :], maxlen=maxlen
        )  # save space for header and footer
    rows.append(values)

    start_live()
    table = create_table(keys, rows, styles)
    live.update(table)  # disabled autorefresh
    return table


def build_parser(d: dict, parser: argparse.ArgumentParser = None, **kwargs):
    """
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    """
    if "description" in kwargs:
        kwargs["description"] = markup_to_ansi(green(kwargs["description"]))

    if parser is None:
        parser = argparse.ArgumentParser(
            formatter_class=lambda prog: argparse.RawTextHelpFormatter(
                prog, max_help_position=140
            ),
            **kwargs,
        )

    help_pattern = f'default = {blue("{}")}'

    for k, v in d.items():
        if isinstance(v, dict):
            if "default" in v:
                # Use other params as kwargs
                d = v.pop("default")
                t = v.pop("type", type(d))
                # h = v.pop('help', markup_to_ansi(help_pattern.format(d)))
                h = (
                    f'{markup_to_ansi(help_pattern.format(d))}; {v.pop("help")}'
                    if "help" in v
                    else markup_to_ansi(help_pattern.format(d))
                )
                parser.add_argument(f"--{k}", default=d, type=t, help=h, **v)
            else:
                parser.add_argument(f"--{k}", **v)

        elif isinstance(v, list):
            parser.add_argument(
                f"--{k}",
                type=type(v[0]) if len(v) else str,
                default=v,
                nargs="+",
                help=markup_to_ansi(help_pattern.format(v)),
            )
        elif isinstance(v, bool):
            parser.add_argument(
                f"--{k}" if not v else f"--no_{k}",
                action="store_true" if not v else "store_false",
                dest=k,
                # default=not v,
                help=markup_to_ansi(help_pattern.format(False)),
            )
            # parser.add_argument(
            #     f"--no_{k}" if not v else f"--{k}",
            #     action="store_false" if not v else "store_true",
            #     dest=k,
            #     # default=v,
            #     help=markup_to_ansi(help_pattern.format(False)),
            # )
        else:
            parser.add_argument(
                f"--{k}",
                type=type(v),
                default=v,
                help=markup_to_ansi(help_pattern.format(v)),
            )

    return parser


def warn_once(message: str):
    if not hasattr(warn_once, "warned"):
        warn_once.warned = set()
    if message not in warn_once.warned:
        log(red(message))
        warn_once.warned.add(message)


def patch_evc_args():
    """
    Call this before importing anything from easyvolcap.engine (including all registered modules) to use your own argument parser
    """
    sep_ind = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    our_args = sys.argv[1:sep_ind]
    evc_args = sys.argv[sep_ind + 1 :]
    sys.argv = (
        [sys.argv[0]] + ["-t", "train"] + evc_args + []
    )  # disable log and use custom logging mechanism
    return our_args


def define_evc_args():
    """
    Before importing anything from the cfg object, define this to indicate that we're an evc program
    and not just using evc as modules
    """
    os.environ["EASYVOLCAP_PROGRAM"] = "1"


def check_evc_args():
    EASYVOLCAP_PROGRAM = os.environ.get("EASYVOLCAP_PROGRAM", "0")
    return (
        EASYVOLCAP_PROGRAM.lower() != "true"
        and EASYVOLCAP_PROGRAM != "0"
        and EASYVOLCAP_PROGRAM
    )


def parse_multilevel_array(string, *args, **kwargs):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid multi-level array input: {string}")


def type_if_not_none(x: Optional[int], type=int):
    if isinstance(x, str) and x.lower() == "none":
        return None
    if x is not None:
        try:
            return type(x)
        except Exception as e:
            log(red(e))
            return x


def str_if_not_none(x: Optional[int]):
    return type_if_not_none(x, str)


def int_if_not_none(x: Optional[int]):
    return type_if_not_none(x, int)


def float_if_not_none(x: Optional[int]):
    return type_if_not_none(x, float)


# Example regular scripts importing utility functions from EasyVolcap
'''
"""
Docstring describing the purpose of the script
"""
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(

    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))


if __name__ == '__main__':
    main()
'''


# Example script invoking easyvolcap programatically, use -- to separate script args and evc args (like configs)
'''
"""
Docstring describing the purpose of the script
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import load_pretrained

from easyvolcap.engine import cfg
from easyvolcap.scripts.main import train, test, gui # will do everything a normal user would do
from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
from easyvolcap.runners.custom_viewer import Viewer
# fmt: on

@catch_throw
def main():
    args = dotdict(
        # Place custom args here like abc=123 and we'll make the default values 123 for args.abc
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    # If you're invoking EasyVolcap, run this to get all initializations, note this requires you to pass in at least a basic config
    # For example, at least do this: python scripts/xxx/xxx.py -- -c configs/exps/xxx/xxx.yaml
    runner = train(cfg, dry_run=True)
    runner.load_network()
    runner.model.network...
    runner.dataloader.dataset...
    ...

    # Otherwise just use your custom modules here like
    viewer = Viewer()
    viewer.run()
    ...


if __name__ == '__main__':
    main()
'''
