# Decrypting Ciphertext Using Markov Chain Monte Carlo  
## Installation

``` bash
cd ../repo_path #after cloning this repo
pyenv virtualenv mcmc_decrypton
pyenv activate mcmc_decrypton
pip install -r requirements.txt


```

## Data
- To create the unigram and bigram transition matrix, we used a chunk of values from the Hebrew wikipedia.
 You can "train" the MCMC process with any data.  
   There is not much Hebrew text on the Internet, but there is some great resources in [NNLP-IL](https://github.com/NNLP-IL/Resources).  
   We used the [MILA](https://yeda.cs.technion.ac.il/index.html) resource .(מרכז ידע לתקשוב בשפה העברית)  
     
 - The program will use the desired text file and will also save a `pickle` file, so you can use your transition matrix later.


- Here is a quick guide to use any of the corpuses of MILA: 
	- Navigate to your project directory: 
	  >      cd .../project_directory 
     -  Download to the file which contains the text data of the relevant corpus (some of the files are .tar and not .zip):  
		  >      wget -c https://yeda.cs.technion.ac.il:8443/corpus/software/corpora/HeWiki_2013/plain/plain.zip | unzip -d ../project_directory
	- Concatenate all the corpus to one big text file: 
	  >      find ../project_directory/zip_folder_name -type f -name '*.txt' -exec cat {} + > corpus_merged.txt
	- Most of the corpuses are very big, so if you wish to use a chunk of the code:  
	  >      head -1000000  corpus_merged.txt >../project_directory/text_files/corpus_merged_1000000.txt`  

This script uses a Markov chain Monte Carlo (MCMC) algorithm to decrypt a message.

## Usage

```python
python decrypt_message.py --seed=<seed> --language=<language> --text_file=<text_file> --message=<message> --plot=<plot> --iterations=<iterations>
```

The following are the arguments that you can pass to the script:

* `--seed`: The random seed.
* `--language`: The language of the message (either "hebrew" or "english").
* `--text_file`: The path to the text file that contains the plaintext.
* `--message`: The encrypted message.
* `--plot`: Whether to plot the results.
* `--iterations`: The number of iterations for the MCMC algorithm.

For example, to decrypt a message in Hebrew using the text file `war-and-peace.txt` with a seed of 42 and 10,000 iterations, you would use the following command:

```python
python decrypt_message.py --seed=42 --language=hebrew --text_file=war-and-peace.txt --message=<message> --plot=False --iterations=10000
```

## Output

The script will output the decrypted message to the console. If you have enabled plotting, the script will also create a plot of the MCMC results.
