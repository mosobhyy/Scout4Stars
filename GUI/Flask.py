import os

from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.utils import secure_filename

from scripts.agility import agility_measure
from scripts.speed import speed_measure
from scripts.power import power_measure

from scripts.df_to_text import *
from scripts.player_classifier import *

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
        try:
            result = next(measures_map.get(int(checkbox_id)))
        except:
            result = 100

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


def players_classifier():
    # Retrieve the uploaded files and selected checkboxes from session variables
    uploaded_files = session.get('uploaded_files')
    selected_checkboxes = session.get('selected_checkboxes')

    # Check if the uploaded files and selected checkboxes exist in the session
    if not uploaded_files or not selected_checkboxes:
        return jsonify({})  # Return an empty dictionary if no progress information is available
    
    # Check if only 1 video uploaded
    if len(uploaded_files) == 1:
        checkbox_id, filename = uploaded_files.popitem()
        # Get path of the saved csv file
        CSV_SAVED_NAME = os.path.basename(filename).split('.')[0]+'_stats.csv'
        CSV_SAVED_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), CSV_SAVED_NAME)

        # Convert DataFrame ro text file for each player
        final_df = initialize_final_df(CSV_SAVED_PATH)
        final_df = fill_column(CSV_SAVED_PATH, final_df, checkbox_id)
        convert_df_to_txt(final_df)
        
    else:
        first_checkbox_id, first_filename= next(iter(uploaded_files.items()))
        classifier = Classifier()
        for checkbox_id, filename in uploaded_files.items():
            if checkbox_id == first_checkbox_id:
                TRAIN_PLAYERS_SAVED_NAME = os.path.basename(filename).split('.')[0]+'_players'
                TRAIN_FOLDER_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), TRAIN_PLAYERS_SAVED_NAME)
                model = classifier.train(TRAIN_FOLDER_PATH)

                # Get csv file of this test 
                CSV_SAVED_NAME = os.path.basename(filename).split('.')[0]+'_stats.csv'
                CSV_SAVED_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), CSV_SAVED_NAME)
                # Initialize final dataframe 
                final_df = initialize_final_df(CSV_SAVED_PATH)
                final_df = fill_column(CSV_SAVED_PATH, final_df, checkbox_id)
            else:
                TEST_PLAYERS_SAVED_NAME = os.path.basename(filename).split('.')[0]+'_players'
                TEST_FOLDER_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), TEST_PLAYERS_SAVED_NAME)
                players_index = classifier.classify(TEST_FOLDER_PATH, TRAIN_FOLDER_PATH, model)

                # Get csv file of this test 
                CSV_SAVED_NAME = os.path.basename(filename).split('.')[0]+'_stats.csv'
                TEST_CSV_SAVED_PATH = os.path.join(os.path.join(os.path.abspath('.'), 'data'), CSV_SAVED_NAME)
                final_df = assign_test_scores_to_players(final_df, players_index, checkbox_id, TEST_CSV_SAVED_PATH)
                
        convert_df_to_txt(final_df)

@skills_app.route("/result")
# Function to render result page
def show_result_page():
    return render_template("ResultReport.html", 
                           page_title = "Result Page",
                           custom_css = 'ResultReportStyle',
                           len = len,
                           )

@skills_app.route("/final_result")
def show_files():
    file_path = 'data'  # Path to the directory containing the files
    file_list = os.listdir(file_path)
    all_files = []
    
    players_classifier()
    for file_name in file_list:
            if os.path.splitext(file_name)[1] == '.txt':
                with open(os.path.join(file_path , file_name), 'r') as file:
                    content = file.readlines()
                    lines = [float(line.strip()) for line in content]
                    all_files.append(lines)

    return render_template("ResultReport.html", 
                           page_title = "Result Page",
                           custom_css = 'ResultReportStyle',
                           data = all_files, len = len)

if __name__ == "__main__":
    skills_app.config["TEMPLATES_AUTO_RELOAD"] = True
    skills_app.run()