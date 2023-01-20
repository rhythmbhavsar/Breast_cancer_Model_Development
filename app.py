from flask import Flask, request, jsonify
import pickle
import numpy as np
import sklearn

model = pickle.load(open('Breast_cancer_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return "Breast Cancer Prediction Machine Learning model by Rhythm Bhavsar"


@app.route('/predict', methods=['POST'])
def predict():
    mean_radius = float(request.form.get('mean_radius'))
    mean_texture = float(request.form.get('mean_texture'))
    mean_perimeter = float(request.form.get('mean_perimeter'))
    mean_area = float(request.form.get('mean_area'))
    mean_smoothness = float(request.form.get('mean_smoothness'))
    mean_compactness = float(request.form.get('mean_compactness'))
    mean_concavity = float(request.form.get('mean_concavity'))
    mean_concave_points = float(request.form.get('mean_concave_points'))
    mean_symmetry = float(request.form.get('mean_symmetry'))
    mean_fractal_dimension = float(request.form.get('mean_fractal_dimension'))
    radius_error = float(request.form.get('radius_error'))
    texture_error = float(request.form.get('texture_error'))
    perimeter_error = float(request.form.get('perimeter_error'))
    area_error = float(request.form.get('area_error'))
    smoothness_error = float(request.form.get('smoothness_error'))
    compactness_error = float(request.form.get('compactness_error'))
    concavity_error = float(request.form.get('concavity_error'))
    concave_points_error = float(request.form.get('concave_points_error'))
    symmetry_error = float(request.form.get('symmetry_error'))
    fractal_dimension_error = float(request.form.get('fractal_dimension_error'))
    worst_radius = float(request.form.get('worst_radius'))
    worst_texture = float(request.form.get('worst_texture'))
    worst_perimeter = float(request.form.get('worst_perimeter'))
    worst_area = float(request.form.get('worst_area'))
    worst_smoothness = float(request.form.get('worst_smoothness'))
    worst_compactness = float(request.form.get('worst_compactness'))
    worst_concavity = float(request.form.get('worst_concavity'))
    worst_concave_points = float(request.form.get('worst_concave_points'))
    worst_symmetry = float(request.form.get('worst_symmetry'))
    worst_fractal_dimension = float(request.form.get('worst_fractal_dimension'))

    input_query = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                             mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                             mean_fractal_dimension, radius_error, texture_error, perimeter_error,
                             area_error, smoothness_error, compactness_error, concavity_error,
                             concave_points_error, symmetry_error, fractal_dimension_error,
                             worst_radius, worst_texture, worst_perimeter, worst_area,
                             worst_smoothness, worst_compactness, worst_concavity,
                             worst_concave_points, worst_symmetry, worst_fractal_dimension]])

    # result = {'mean_radius': mean_radius, 'mean_texture': mean_texture, 'mean_perimeter': mean_perimeter,
    #           'mean_area': mean_area, 'mean_smoothness': mean_smoothness, 'mean_compactness': mean_compactness,
    #           'mean_concavity': mean_concavity, 'mean_concave_points': mean_concave_points,
    #           'mean_symmetry': mean_symmetry, 'mean_fractal_dimension': mean_fractal_dimension,
    #           'radius_error': radius_error, 'texture_error': texture_error,
    #           'perimeter_error': perimeter_error, 'area_error': area_error,
    #           'smoothness_error': smoothness_error, 'compactness_error': compactness_error,
    #           'concavity_error': concavity_error, 'concave_points_error': concave_points_error,
    #           'symmetry_error': symmetry_error, 'fractal_dimension_error': fractal_dimension_error,
    #           'worst_radius': worst_radius, 'worst_texture': worst_texture,
    #           'worst_perimeter': worst_perimeter, 'worst_area': worst_area,
    #           'worst_smoothness': worst_smoothness, 'worst_compactness': worst_compactness,
    #           'worst_concavity': worst_concavity, 'worst_concave_points': worst_concave_points,
    #           'worst_symmetry': worst_symmetry, 'worst_fractal_dimension': worst_fractal_dimension}

    result = model.predict(input_query)[0]

    return jsonify({'Outcome': str(result)})
    # return str(result)


# if __name__ == '__ main__':
#     app.run(debug=True, port=80)

if __name__ == '__main__':
    app.run(debug=True)
# app.run(debug=True, port=80)
