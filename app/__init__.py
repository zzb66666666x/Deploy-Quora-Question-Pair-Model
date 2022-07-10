"""Setup at app startup"""
import os
import sys
from flask import Flask
from yaml import load, Loader

app = Flask(__name__)

# conn = db.connect()
# results = conn.execute("Select * from Category")
# we do this because results is an object, this is just a quick way to verify the content
# print([x for x in results])
# conn.close()

# To prevent from using a blueprint, we use a cyclic import
# This also means that we need to place this import here
# pylint: disable=cyclic-import, wrong-import-position
from app import routes