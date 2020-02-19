#!/usr/bin/env bash

ipython nbconvert --to html document.ipynb && wkhtmltopdf -T 0 -B 0 --page-width 210mm --page-height 455mm document.html document.pdf
rm document.html
