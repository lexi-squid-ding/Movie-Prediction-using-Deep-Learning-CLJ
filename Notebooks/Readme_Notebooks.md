NOTEBOOKS/SCRIPTS: ZIPPED INTO LARGE FOLDER (Also accessible on GitHub)

**MILESTONE 1:**
- Milestone_1_summary: final submission for Milestone 1, performs Exploratory Data Analysis of features by genres, including the results of all the notebooks below.
- Cynthia Project Note.ipynb: Downloads a small subset of movies (top 100 of all time) using the TMDB and IMDB, explores correlation matrix between genres (pairwise counts of movies) and generates correlation matrix figure. Explores movie languages and creates dictionary of movie genres by TMDB ID
- Exploratory Data Analysis (Lexi).ipynb: Loads datasets and prints out values to look for encoding errors
- Milestone 1 Lexi.ipynb; Downloads poster of a particular movie and examines metadata and poster sizes, compares genre for a small subset of movies in the two databases (TMDB, IMDB), uses TMDB API to get top 10 movies and fix formatting errors in label encoding. Visualizes the number of genres per movie, release month, genres frequency in the top 100 movies of 2016.
- Exploratory_JY.ipynb: Preliminary text analysis of movie titles and overviews. Identifies popular words, average title lengths by genre. Looks at trends in popularity and number of votes by genre.
- Try_Downloading_Movie_genres.ipynb: Troubleshoots encoding errors caused by non-utf-8 encoding of foreign characters (ex: French words).
- Additional Extensions: Did exploratory data analysis using Principal Components Plots and clustering in R (see milestone 5)
 
**MILESTONE 2:**
- EDA of Data from 2011-2016.ipynb: Creates EDA plots for data downloaded from 2011-2016 (number of genres per movie, number of movies per genre, correlation matrix).
- CombineYearData.ipynb: runs loop to download top 1000 movies from 2012-2016, combine data from multiple years
- Download Full Dataset_top1000 or 2000.ipynb: Runs loop to download movie metadata using TMDB API, and incorporates some mechanisms to bypass encoding errors, save data to CSV.
- IMDb.ipynb: Experimenting with IMDB API to get the directors, cast and crew for each movie.
- Milestone_2_summary: Describes data, selection of feature and response variable, Label encodes (binary encoding) genre labels and turning into one hot encoding, Addresses ways to combat class imbalance and future directions for text, image analysis.
- Additional Extensions: wrote extensive code to fix encoding errors in imdb text, and test the script’s performance for problematic data entries (largely foreign films)- in folder fix_imdb_encoding
 
**MILESTONE 3:**
- Milestone_3_summary.ipynb: Summary of milestone 3 (Traditional Statistical Learning Methods), visualizations/tables of relative accuracy
- data_cleaning.py/ Text Analysis.ipynb: Loads corpus of text data from the titles and overviews; examines frequency of words, and also cleans up encoding errors caused by non-utf characters in foreign languages. Performs exploratory PCA on word counts and also extracts quantitative response variables.
- format_runtime_aspectratio.py: formats the runtime and aspect ratio, cleaning rows with invalid entries.
- Assess Model Accuracy – LogReg and basic Tree and Random Forest.ipynb: Creates and tunes, via Grid Search Cross-Validation, Logistic Regression, Decision Tree, Random Forest, and AdaBoost models for genre prediction. Saves outputs to pickle files, and then loads and evaluates performance metrics.
- Loads files created by Test_finetune,ipynb and evaluates metrics to assess model accuracy
- Additional Extensions: Rerun the models on updated dataset with cleaned movie list and genre list. Performed SVM with linear, polynomial and RBF kernels on the data. 
 
**MILESTONE 4:**
- Data_preparation_genre_Lexi.ipynb: Preliminary genre predictions from scratch, runs simple 1 layer neural net from posters in 2011-2016, examines baseline model, Creates 2 layer net for comparison with 1 layer net, and experiments with basic changes to network architecture
- Download Poster URLs from Top 1000 movies of each year from 2005-2010 (6 years).ipynb: Scrapes poster URL data from earlier time period 2005-2010.
- Finetune_genre.ipynb: Fine-tuning the VGG16 pre-trained model to predict genre models, did not achieve high accuracy but experimented with different methods and assessed model performances. 
- FineTune_PretrainedCNN_Face.ipynb: Using finetuned VGG16 model to predict hte number of faces, based on manaully scored gold standard. 
- NNet_JY.ipynb: Preliminary exploration of neural nets to predict 
- Milestone_4_Sumary_1,2,3.ipynb (3 files): Milestone 4 summary (in 3 separate files)
- get_list_poster_names.py: Small script to get the list of Movie IDs with posters
- Additional Extensions: In this milestone, in addition to trying to predict genre using the fine-tuned and from-scratch models, we also MANUALLY SCORED the number of faces and the title position in movie posters, and trained neural nets on the task of predicting nfaces and title position. The reason why we did this was because we felt that if posters of different genres differ in the number of faces or title position, we might be able to add these features to the genre classification and improve accuracy. In addition, starting from simpler features such as title position and face numbers might be a easier and possible task for built-from-scratch models. 

 
**MILESTONE 5:**
- Finetune_genre_animation.ipynb: Fine-tunes the VGG16 pre-trained model to predict animation label
- Finetune_genre_drama.ipynb: Fine-tunes the VGG16 pre-trained model to predict drama label.
- Neural_Nets_scratch_Lexi.ipynb: Tunes neural nets from scratch, experimenting with different depths (n layers), dimensionality (n nodes), and dense layer architectures. Tries to overfit small neural nets to data and then experiments with deeper nets (16 layer) from scratch.
- Regroup_Genres.ipynb: Preliminary exploration of the idea of regrouping the 20 genres into closely related categories to reduce the extent of the multi-label problem. We did not end up using the labels generated by this script. 
