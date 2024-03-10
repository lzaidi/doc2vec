# Article Recommendation System

This Python script implements a simple article recommendation system based on GloVe word vectors and Euclidean distances. It processes a set of articles, computes their word vector centroids, and provides recommendations for articles based on their similarity.

## Usage

1. **Clone Repository:**
    ```bash
    git clone https://github.com/lzaidi/article-recommendation.git
    ```

2. **Navigate to the Project Directory:**
    ```bash
    cd article-recommendation
    ```

3. **Install Dependencies:**
    ```bash
    pip install numpy flask
    ```

4. **Run the Recommendation System:**
    ```bash
    python app.py glove_file_path articles_directory_path
    ```

   - `glove_file_path`: Path to the GloVe word vector file.
   - `articles_directory_path`: Path to the directory containing article text files.

5. **Access the Web Interface:**
   Open your web browser and navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to explore the article recommendations.

## Project Structure

- `app.py`: Flask web application for serving article recommendations.
- `recommend.py`: Python script for recommending articles.
- `templates/`: HTML templates for rendering web pages.
- `static/`: CSS and JavaScript files for styling the web interface.
- `glove/`: Directory containing GloVe word vector files.
- `articles/`: Directory containing text files of articles.

## Dependencies

- Flask: Web application framework for Python.
- Numpy: Numerical computing library for Python.

## File Descriptions

- `app.py`: Flask web application script.
- `recommend.py`: Recommendation system script.
- `templates/`: HTML templates for rendering web pages.
- `static/`: CSS and JavaScript files.
- `glove/`: Directory for storing GloVe word vector files.
- `articles/`: Directory for storing text files of articles.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
