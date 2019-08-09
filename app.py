# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 02:04:51 2019

@author: SRIJAN
"""

from flask import Flask, render_template      

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")
    

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)