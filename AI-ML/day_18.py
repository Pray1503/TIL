# Day 18 - Machine Learning (Improved Flask UI)

# What I Learned:
# Machine learning web applications can be improved using HTML and CSS
# to create a more professional and user-friendly interface.

# Key Concept:
# Frontend styling enhances user experience and makes ML applications
# look more interactive and visually appealing.

# UI Improvements:
# - Better layout
# - Styled buttons
# - Background colors
# - Responsive form design
# - Prediction display card

# Important Points:
# - Flask can render HTML templates
# - CSS is used for styling web pages
# - Good UI improves usability
# - User-friendly apps are important in real-world deployment

# Conclusion:
# Improving the frontend of machine learning applications helps create
# professional and interactive AI-powered web products.

# Day 18 - Improved Flask UI for ML Prediction

from flask import Flask, request, render_template_string
import joblib

# Load trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

# HTML + CSS
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Predictor</title>

    <style>
        body{
            font-family: Arial;
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
        }

        .container{
            background:white;
            padding:30px;
            border-radius:15px;
            width:400px;
            box-shadow:0px 0px 20px rgba(0,0,0,0.2);
        }

        h1{
            text-align:center;
            color:#333;
        }

        input{
            width:100%;
            padding:10px;
            margin-top:10px;
            border-radius:8px;
            border:1px solid #ccc;
        }

        button{
            width:100%;
            padding:12px;
            margin-top:20px;
            border:none;
            background:#4CAF50;
            color:white;
            font-size:16px;
            border-radius:8px;
            cursor:pointer;
        }

        button:hover{
            background:#45a049;
        }

        .result{
            margin-top:20px;
            padding:15px;
            background:#f0f0f0;
            border-radius:10px;
            text-align:center;
            font-size:18px;
            font-weight:bold;
        }
    </style>
</head>

<body>

<div class="container">

    <h1>🌸 Iris Flower Predictor</h1>

    <form method="POST">

        <input type="text" name="sl" placeholder="Sepal Length">

        <input type="text" name="sw" placeholder="Sepal Width">

        <input type="text" name="pl" placeholder="Petal Length">

        <input type="text" name="pw" placeholder="Petal Width">

        <button type="submit">Predict Flower</button>

    </form>

    {% if prediction %}
        <div class="result">
            Prediction: {{ prediction }}
        </div>
    {% endif %}

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        try:
            sl = float(request.form["sl"])
            sw = float(request.form["sw"])
            pl = float(request.form["pl"])
            pw = float(request.form["pw"])

            features = [[sl, sw, pl, pw]]

            pred = model.predict(features)[0]

            classes = ["Setosa", "Versicolor", "Virginica"]

            prediction = classes[pred]

        except:
            prediction = "Invalid Input"

    return render_template_string(html, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
