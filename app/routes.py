""" Specifies routing for the application"""
from flask import render_template, request, jsonify, redirect, make_response
from app import app
from app import predict as pred

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def make_prediction():
    data = request.get_json()
    q1 = data["question1"]
    q2 = data["question2"]
    print(len(q1))
    print(len(q2))
    if q1 == "" or q2 == "":
        result = {"success": True, "response": "please enter two questions to check"}
    else:
        try: 
            ret = pred.get_prediction(q1, q2)
            result = {"success": True, "response": "prediction made: " + ret}
        except:
            result = {"success": False, "response": "cannot predict"}
    return jsonify(result)

