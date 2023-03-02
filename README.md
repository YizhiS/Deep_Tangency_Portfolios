# Deep Tangency Portfolios
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3971274

# Example Data Description
2004.07 - 2020.12 monthly data (198 months)

* "one_month_bill_2020.txt": one month bill return referred as risk-free rate.


## Input 

* "data": a dictionary of all kinds of data. The keys are 
  - "characteristics"
  - "stock_return"
  - "target_return" (i.e. portoflio return)
  - "factor" (i.e. benchmark factors)


* "char_type" : characteristics types.
* "bm_type" : benchmark factors.
* "port_type" : market type, equally weighted or value weighted.

* "layer_size": a list of hidden layer size where the last element is the number of deep factors to be constructed. For example, [32,16,8,4,2].

* "learning_rate": learning rate, a parameter for training algorithm.
* "gamma_A" : tunning parameter for off diagonal penalty.
* "gmma_l2" : tunning parameter for weight matrix regularization.

## Output

* "factor": deep factors
* "weights": individual assets' weights
* "dchar": deep characteristics
