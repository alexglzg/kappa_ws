"""kappa_experiments package.

On import, adds <ws_root>/src/kappa-motion-planner/src to sys.path so that
``import kappa_planner`` works when the library is only checked out into the
workspace (same mechanism as the corridor_planner package). If kappa_planner is
pip-installed this is a no-op.
"""
import os
import sys


def _link_kappa_planner():
    try:
        from ament_index_python.packages import get_package_prefix
        pkg_install_dir = get_package_prefix('kappa_experiments')
        ws_root = os.path.abspath(os.path.join(pkg_install_dir, '..', '..'))
        kappa_path = os.path.join(ws_root, 'src', 'kappa-motion-planner', 'src')
        if os.path.isdir(kappa_path) and kappa_path not in sys.path:
            sys.path.insert(0, kappa_path)
    except Exception as exc:  # pragma: no cover
        print(f'[kappa_experiments] could not link kappa_planner automatically: {exc}')


_link_kappa_planner()
