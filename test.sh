#!/usr/bin/env bash

python3 test1.py --checkpoint model/SR/best_model.pth &
python3 test2.py --checkpoint model/SR/best_model.pth &
python3 test3.py --checkpoint model/SR/best_model.pth &
python3 test4.py --checkpoint model/SR/best_model.pth
