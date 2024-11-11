# Preprocessing automation with Qwen2.5 LLM.

This repository contains technical documentation and a web application developed with Flask. The documentation explores various chapters of a study (Introduction, Qwen2.5, Application, Results, Conclusions), while the web application provides an interface for visualizing and processing the described data.

## Repository Structure.

- **documentation/**: Contains documentation in LaTeX and PDF format, divided into chapters and sections.
  - `chapters/`: `.tex` files of individual chapters (Introduction, Qwen2, Application, Results, Conclusions).
  - `media/`: Images and PDF files used in the documentation.
  - `slides/`: PDF and LaTeX presentations related to the documentation.

- **python/flask_app/**: Web application developed with Flask.
  - `app.py`: Main file that starts the Flask app.
  - `generated_preprocessing.py`: Data preprocessing script.
  - `qwen2_service.py` and `service.py`: Service modules for the application.
  - `static/`: Contains CSS, JavaScript and image files used in the app.
  - `templates/`: Contains the HTML files used as templates for the app.

## Requirements.

- **Python 3.x**
- **Flask**: Installable via `pip install flask`.

## Instructions for Execution.

1. Clone the repository:

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask application:
   ```bash
   python python/flask_app/app.py
   ```

4. Access the application through the browser at [http://localhost:5000](http://localhost:5000).

## License.

This project is released under the MIT license. 

Copyright (c) [2024].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and related documentation files (the “Software”), to use the Software without restriction, including, without limitation, the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do the same, subject to the following conditions:

The above copyright notice and this permission notice must be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS,” WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGE OR OTHER LIABILITY, WHETHER IN AN ACTION IN CONTRACT, TORT OR OTHERWISE, ARISING OUT OF, OR IN CONNECTION WITH, THE SOFTWARE OR THE USE OF OR OTHER ACTIVITIES IN THE SOFTWARE.
