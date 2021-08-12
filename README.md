# mtg_ml_models

## Introduction
This project is a continuation of mtg_scraper project

This project consists of both classification and regression analysis, for which I applied supervised learning based on data previously collected with my scraper. 


## Project Brief & Deliverables

The project brief was to:
* Identify an industry relevant prediction problem.
* Develop a solution to this problem.
* Present the results.

Deliverables:
* A GitHub repo containing all code.
* A presentation in two parts:
    1. **Non-technical presentation** highlighting the problem and the approach to a solution.
    2. **Technical presentation** giving details of the techniques applied during data processing and modelling.


### Stakeholders

Stakeholders for this project include:
- The company owning the game: 
The plan was to check for the “fairness” and randomisation of the game to see if there are any biases that can lead to unbalanced gaming which -eventually- can lead to a worse experience for the players and a decline in the interest for the game.
- Myself:  
I also checked for a correlation between the values of the cards in the secondary market that could potentially lead to identification of under or over valued cards and the profit potential that could derive.


## Data
### Data Cleaning

Data cleaning was performed mostly as part of the mtg_scraper. Additional cleaning was needed in order to drop outlier values or features that had no intrinsic value to the specific project

 

## Modelling

For the classification analysis, 14 different models were trained and compared with a baseline

For the regressions analysis, I only run a Linear regression to get a benchmark.

## Work in progress

Classification models need to be enriched with more parametrisation to check if the results can be improved

Regression analysis needs more models so that a proper comparison between models can be performed and better results can be acquired.

main_cat and main_lin_regr files to be merged to one.
