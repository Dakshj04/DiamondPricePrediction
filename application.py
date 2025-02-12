from flask import Flask, render_template, request, jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
application = Flask(__name__)
app=application
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET",'POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
        data = CustomData(
            carat=float(request.form['carat']),
            cut=request.form['cut'],
            color=request.form['color'],
            clarity=request.form['clarity'],
            depth=float(request.form['depth']),
            table=float(request.form['table']),
            x=float(request.form['x']),
            y=float(request.form['y']),
            z=float(request.form['z'])
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(final_new_data)
        results=round(prediction[0],2)
        return render_template("form.html",final_result=results)
     
if __name__ == '__main__':
    app.run(debug=True)