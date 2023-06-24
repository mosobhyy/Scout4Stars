import os
from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.utils import secure_filename

skills_app = Flask(__name__)
# Generate a secret key
secret_key = os.urandom(24)
skills_app.secret_key = secret_key
skills_app.config['UPLOAD_FOLDER'] = r'C:\Users\pc\Desktop\Korastat_GUI'
skills_app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv'}  # Set the allowed file extensions


my_skills = [("Speed", 60), ("Dribble", 20), ("Agility", 40), ("Shoot", 90), ("Pass", 10), ("Power", 80),]

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
                    file.save(os.path.join(skills_app .config['UPLOAD_FOLDER'], filename))

                    # Associate the uploaded file with the relevant checkbox
                    if checkbox not in uploaded_files:
                        uploaded_files[checkbox] = []

                    uploaded_files[checkbox].append(filename)

                # Store the uploaded files and selected checkboxes in session variables
                session['uploaded_files'] = uploaded_files
                session['selected_checkboxes'] = selected_checkboxes

                # Perform further processing with the uploaded files and selected checkboxes
                print(selected_checkboxes)
                print(uploaded_files)

                return redirect('/progress')
   return render_template("skills page.html",
                            page_title="Skills Page",
                            custom_css='skills',
                            message='')



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
    for checkbox, files in uploaded_files.items():
        for file in files:
            # Calculate the progress value based on the current frame movement progress
            progress_value = session.get(file + '_progress')

            # Add the progress value to the progress dictionary
            progress[file] = progress_value

    return render_template("progress page.html",
                           page_title="Progress Page",
                           custom_css='progress',
                           page_head='Videos Progress',
                           data=my_skills,
                           progress=progress)


@skills_app.route("/result")
def show_files():
    file_path = 'data'  # Path to the directory containing the files
    file_list = os.listdir(file_path)
    file_contents = []

    for file_name in file_list:
        if file_name.endswith('.txt'):  # Filter only text files
            file_base_name = os.path.splitext(file_name)[0]
            with open(os.path.join(file_path, file_name), 'r') as file:
                content = file.readlines()
                file_contents.append((file_name, content))
    return render_template("ResultReport.html", 
                           page_title="Result Page",
                           custom_css='ResultReportStyle',
                           files=file_contents)


                           

if __name__ == "__main__":
    skills_app.config["TEMPLATES_AUTO_RELOAD"] = True
    skills_app.run(debug=True)