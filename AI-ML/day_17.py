# Day 17 - Machine Learning (Flask Web App)

# What I Learned:
# Machine learning models can be deployed using web frameworks like Flask
# to create user-friendly applications.

# Key Concept:
# Flask allows us to build a simple web interface where users can input
# data and receive predictions from a trained model.

# Workflow:
# - Load trained model
# - Create web interface (HTML form)
# - Accept user input
# - Make prediction
# - Display result

# Important Points:
# - Flask is lightweight and easy to use
# - Enables real-time interaction with ML models
# - Forms are used to collect user input
# - Models can be integrated into web apps

# Conclusion:
# Flask helps in deploying machine learning models as interactive web
# applications, making them accessible to users.

# Day 17 - Flask Web App for ML Model

from flask import Flask, request, render_template_string
import joblib

# Load trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

# HTML Template
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Flower Predictor</title>
</head>
<body>
    <h2>🌸 Iris Flower Prediction</h2>

    <form method="POST">
        Sepal Length: <input type="text" name="sl"><br><br>
        Sepal Width: <input type="text" name="sw"><br><br>
        Petal Length: <input type="text" name="pl"><br><br>
        Petal Width: <input type="text" name="pw"><br><br>
        <input type="submit" value="Predict">
    </form>

    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
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
            prediction = "Invalid input. Please enter numbers."

    return render_template_string(html, prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
