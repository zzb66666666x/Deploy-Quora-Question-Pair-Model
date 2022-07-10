#!/bin/zsh
cd "$(dirname $0)"
export FLASK_APP=app # this tells the terminal that our flask app module is in the app folder
export FLASK_DEBUG=1 # this tells flask to reload server on change
flask run
