"""Public V2 Studio entry; widgets and workers live in studio/."""
import argparse
import json
from pathlib import Path
import sys
from .qt_runtime import prepare_qt
prepare_qt()
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from config import AgentConfig
from studio.window import Studio
from studio.runtime import AgentWorker


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', default='把蓝色盒子放到绿色积木右边')
    parser.add_argument('--config')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--autorun', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--required-llm', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--quit-after-task', action='store_true')
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setFont(QFont('PingFang SC', 12))
    studio = Studio(AgentConfig.load(args.config), args.instruction)
    studio.mode.setCurrentIndex(studio.mode.findData('offline' if args.offline else 'required' if args.required_llm else studio.config.llm.mode))
    if not args.no_show: studio.show()
    if args.autorun:
        def autorun():
            studio.worker.ready.disconnect(autorun)
            QTimer.singleShot(300, studio.run_task)
        studio.worker.ready.connect(autorun)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        save = lambda:studio.grab().save(str(args.screenshot))
        if args.autorun: studio.worker.finished_task.connect(lambda _:QTimer.singleShot(200, save))
        else: studio.worker.ready.connect(lambda:QTimer.singleShot(600, save))
    if args.quit_after_task:
        studio.worker.finished_task.connect(lambda result:print(json.dumps(result, ensure_ascii=False), flush=True))
        studio.worker.finished_task.connect(lambda _:QTimer.singleShot(700, studio.close))
        studio.worker.fatal.connect(lambda _:QTimer.singleShot(300, studio.close))
    if args.smoke: studio.worker.ready.connect(lambda:QTimer.singleShot(2000, studio.close))
    if args.smoke: studio.worker.fatal.connect(lambda _:QTimer.singleShot(300, studio.close))
    return app.exec()


if __name__ == '__main__': raise SystemExit(main())
