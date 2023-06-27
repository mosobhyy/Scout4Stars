import os

from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.utils import secure_filename

from scripts.agility import agility_measure
from scripts.speed import speed_measure
from scripts.power import power_measure

measures_map = {1: speed_measure,
                2: power_measure,
                5: agility_measure
                }

skills_app = Flask(__name__)
# Generate a secret key
secret_key = os.urandom(24)
skills_app.secret_key = secret_key
print(os.path.join(f"{os.path.abspath('.')}", "data"))

# skills_app.config['UPLOAD_FOLDER'] = os.path.join(f"{os.path.abspath('.')}", "data")

skills_app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv'}  # Set the allowed file extensions

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in skills_app.config['ALLOWED_EXTENSIONS']

@skills_app.route("/", methods=['GET', 'POST'])
def skills():
   if request.method == 'POST':
        selected_checkboxes = request.form.getlist('mycheckbox')

        # Check if the run button was clicked
        if 'run' in request.form:
            if not selected_checkboxes:
                # No checkboxes selected
                return render_template("skills page.html", custom_css='skills', message='Please select at least one test.')
            else:
                uploaded_files = {}
                for checkbox in selected_checkboxes:
                    file = request.files.get(f'video_{checkbox}')
                    # Check if the file is selected
                    if file.filename == '':
                        return 'No file selected.'
                    
                    # Check if the file has an allowed extension
                    if not allowed_file(file.filename):
                        return 'Invalid file extension.'

                    # Save the uploaded file to a desired location
                    filename = secure_filename(file.filename)
                    # file.save(os.path.join(skills_app.config['UPLOAD_FOLDER'], filename))

                    uploaded_files[checkbox] = filename

                # Store the uploaded files and selected checkboxes in session variables
                session['uploaded_files'] = uploaded_files
                session['selected_checkboxes'] = selected_checkboxes

                return redirect('/progress')
   return render_template("skills page.html",
                            page_title="Skills Page",
                            custom_css='skills',
                            message='')

@skills_app.route("/progress_status")
def progress_status():
    uploaded_files = session.get('uploaded_files')

    progress_list = {}

    # Call the next iteration for every function
    for checkbox_id in uploaded_files.keys():
        # try:
            # result = next(measures_map.get(int(checkbox_id)))
        # except:
            # result = 100
        result = next(measures_map.get(int(checkbox_id)))
        

        progress_list[int(checkbox_id)-1] = result

    return progress_list

@skills_app.route("/progress")
def progress():
    # Retrieve the uploaded files and selected checkboxes from session variables
    uploaded_files = session.get('uploaded_files')
    selected_checkboxes = session.get('selected_checkboxes')

    # Check if the uploaded files and selected checkboxes exist in the session
    if not uploaded_files or not selected_checkboxes:
        return jsonify({})  # Return an empty dictionary if no progress information is available

    # Retrieve the progress of frame movement from session variables or a database
    progress = {}  # Initialize an empty dictionary for progress

    # Example: Retrieve the progress value for each file from session variables
    for checkbox_id, filename in uploaded_files.items():
        measures_map[int(checkbox_id)] = measures_map[int(checkbox_id)](filename)
        
    return render_template("progress page.html",
                        page_title="Progress Page",
                        custom_css='progress',
                        page_head='Videos Progress',
                        progress=progress)


@skills_app.route("/result")
def show_files():
    file_path = 'data'  # Path to the directory containing the files
    file_list = os.listdir(file_path)
    all_files = []

    for file_name in file_list:
            with open(os.path.join(file_path , file_name), 'r') as file:
                content = file.readlines()
                lines = [float(line.strip()) for line in content]
                all_files.append(lines)

    
    return render_template("ResultReport.html", 
                           page_title="Result Page",
                           custom_css='ResultReportStyle',
                           data=all_files, len = len)

if __name__ == "__main__":
    skills_app.config["TEMPLATES_AUTO_RELOAD"] = True
    skills_app.run(debug=True)