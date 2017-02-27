Machine Learning Engineer Nanodegree[¶](#Machine-Learning-Engineer-Nanodegree) {#Machine-Learning-Engineer-Nanodegree}
==============================================================================

Introduction and Foundations[¶](#Introduction-and-Foundations) {#Introduction-and-Foundations}
--------------------------------------------------------------

Project: Titanic Survival Exploration[¶](#Project:-Titanic-Survival-Exploration) {#Project:-Titanic-Survival-Exploration}
--------------------------------------------------------------------------------

In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and
sank, resulting in the deaths of most of its passengers and crew. In
this introductory project, we will explore a subset of the RMS Titanic
passenger manifest to determine which features best predict whether
someone survived or did not survive. To complete this project, you will
need to implement several conditional predictions and answer the
questions below. Your project submission will be evaluated based on the
completion of the code and your responses to the questions.

> **Tip:** Quoted sections like this will provide helpful instructions
> on how to navigate and use an iPython notebook.

Getting Started[¶](#Getting-Started) {#Getting-Started}
====================================

To begin working with the RMS Titanic passenger data, we'll first need
to `import` the functionality we need, and load our data into a `pandas`
DataFrame.\
 Run the code cell below to load our data and display the first few
entries (passengers) for examination using the `.head()` function.

> **Tip:** You can run a code cell by clicking on the cell and using the
> keyboard shortcut **Shift + Enter** or **Shift + Return**.
> Alternatively, a code cell can be executed using the **Play** button
> in the hotbar after selecting it. Markdown cells (text cells like this
> one) can be edited by double-clicking, and saved using these same
> shortcuts.
> [Markdown](http://daringfireball.net/projects/markdown/syntax) allows
> you to write easy-to-read plain text that can be converted to HTML.

In [154]:

    # Import libraries necessary for this project
    import numpy as np
    import pandas as pd
    from IPython.display import display # Allows the use of display() for DataFrames

    # Import supplementary visualizations code visuals.py
    import visuals as vs

    # Pretty display for notebooks
    %matplotlib inline

    # Load the dataset
    in_file = 'titanic_data.csv'
    full_data = pd.read_csv(in_file)

    # Print the first few entries of the RMS Titanic data
    display(full_data.head())

PassengerId

Survived

Pclass

Name

Sex

Age

SibSp

Parch

Ticket

Fare

Cabin

Embarked

0

1

0

3

Braund, Mr. Owen Harris

male

22.0

1

0

A/5 21171

7.2500

NaN

S

1

2

1

1

Cumings, Mrs. John Bradley (Florence Briggs Th...

female

38.0

1

0

PC 17599

71.2833

C85

C

2

3

1

3

Heikkinen, Miss. Laina

female

26.0

0

0

STON/O2. 3101282

7.9250

NaN

S

3

4

1

1

Futrelle, Mrs. Jacques Heath (Lily May Peel)

female

35.0

1

0

113803

53.1000

C123

S

4

5

0

3

Allen, Mr. William Henry

male

35.0

0

0

373450

8.0500

NaN

S

From a sample of the RMS Titanic data, we can see the various features
present for each passenger on the ship:

-   **Survived**: Outcome of survival (0 = No; 1 = Yes)
-   **Pclass**: Socio-economic class (1 = Upper class; 2 = Middle class;
    3 = Lower class)
-   **Name**: Name of passenger
-   **Sex**: Sex of the passenger
-   **Age**: Age of the passenger (Some entries contain `NaN`)
-   **SibSp**: Number of siblings and spouses of the passenger aboard
-   **Parch**: Number of parents and children of the passenger aboard
-   **Ticket**: Ticket number of the passenger
-   **Fare**: Fare paid by the passenger
-   **Cabin** Cabin number of the passenger (Some entries contain `NaN`)
-   **Embarked**: Port of embarkation of the passenger (C = Cherbourg; Q
    = Queenstown; S = Southampton)

Since we're interested in the outcome of survival for each passenger or
crew member, we can remove the **Survived** feature from this dataset
and store it as its own separate variable `outcomes`. We will use these
outcomes as our prediction targets.\
 Run the code cell below to remove **Survived** as a feature of the
dataset and store it in `outcomes`.

In [139]:

    # Store the 'Survived' feature in a new variable and remove it from the dataset
    outcomes = full_data['Survived']
    data = full_data.drop('Survived', axis = 1)

    # Show the new dataset with 'Survived' removed
    display(data.head())

PassengerId

Pclass

Name

Sex

Age

SibSp

Parch

Ticket

Fare

Cabin

Embarked

0

1

3

Braund, Mr. Owen Harris

male

22.0

1

0

A/5 21171

7.2500

NaN

S

1

2

1

Cumings, Mrs. John Bradley (Florence Briggs Th...

female

38.0

1

0

PC 17599

71.2833

C85

C

2

3

3

Heikkinen, Miss. Laina

female

26.0

0

0

STON/O2. 3101282

7.9250

NaN

S

3

4

1

Futrelle, Mrs. Jacques Heath (Lily May Peel)

female

35.0

1

0

113803

53.1000

C123

S

4

5

3

Allen, Mr. William Henry

male

35.0

0

0

373450

8.0500

NaN

S

The very same sample of the RMS Titanic data now shows the **Survived**
feature removed from the DataFrame. Note that `data` (the passenger
data) and `outcomes` (the outcomes of survival) are now *paired*. That
means for any passenger `data.loc[i]`, they have the survival outcome
`outcomes[i]`.

To measure the performance of our predictions, we need a metric to score
our predictions against the true outcomes of survival. Since we are
interested in how *accurate* our predictions are, we will calculate the
proportion of passengers where our prediction of their survival is
correct. Run the code cell below to create our `accuracy_score` function
and test a prediction on the first five passengers.

**Think:** *Out of the first five passengers, if we predict that all of
them survived, what would you expect the accuracy of our predictions to
be?*

In [140]:

    def accuracy_score(truth, pred):
        """ Returns accuracy score for input truth and predictions. """
        
        # Ensure that the number of predictions matches number of outcomes
        if len(truth) == len(pred): 
            
            # Calculate and return the accuracy as a percent
            return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
        
        else:
            return "Number of predictions does not match number of outcomes!"
        
    # Test the 'accuracy_score' function
    predictions = pd.Series(np.ones(5, dtype = int))
    print accuracy_score(outcomes[:5], predictions)

    Predictions have an accuracy of 60.00%.

> **Tip:** If you save an iPython Notebook, the output from running code
> blocks will also be saved. However, the state of your workspace will
> be reset once a new session is started. Make sure that you run all of
> the code blocks from your previous session to reestablish variables
> and functions before picking up where you last left off.

Making Predictions[¶](#Making-Predictions) {#Making-Predictions}
==========================================

If we were asked to make a prediction about any passenger aboard the RMS
Titanic whom we knew nothing about, then the best prediction we could
make would be that they did not survive. This is because we can assume
that a majority of the passengers (more than 50%) did not survive the
ship sinking.\
 The `predictions_0` function below will always predict that a passenger
did not survive.

In [141]:

    def predictions_0(data):
        """ Model with no features. Always predicts a passenger did not survive. """

        predictions = []
        for _, passenger in data.iterrows():
            
            # Predict the survival of 'passenger'
            predictions.append(0)
        
        # Return our predictions
        return pd.Series(predictions)

    # Make the predictions
    predictions = predictions_0(data)

### Question 1[¶](#Question-1) {#Question-1}

*Using the RMS Titanic data, how accurate would a prediction be that
none of the passengers survived?*\
 **Hint:** Run the code cell below to see the accuracy of this
prediction.

In [142]:

    print accuracy_score(outcomes, predictions)

    Predictions have an accuracy of 61.62%.

**Answer:** Predictions with the assumption that none of the passengers
survived have an accuracy of 61.62%.

* * * * *

Let's take a look at whether the feature **Sex** has any indication of
survival rates among passengers using the `survival_stats` function.
This function is defined in the `titanic_visualizations.py` Python
script included with this project. The first two parameters passed to
the function are the RMS Titanic data and passenger survival outcomes,
respectively. The third parameter indicates which feature we want to
plot survival statistics across.\
 Run the code cell below to plot the survival outcomes of passengers
based on their sex.

In [143]:

    vs.survival_stats(data, outcomes, 'Sex')

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfgAAAGDCAYAAADHzQJ9AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3XmYXHWd7/H3104gCJE1cAMBEhlUCCQBml0wFxxAiUEd%0AIEFkwKsCLpC5OirgAogoI+IdNwbh6pAraAhxxMgyoIyACIIJJmgCTNgkgUBCZAkRMAnf+8c5HSqd%0AXqqX6u6cvF/PU09XnTrLt7b+1O93fnVOZCaSJKla3tDfBUiSpN5nwEuSVEEGvCRJFWTAS5JUQQa8%0AJEkVZMBLklRBBrzUhyLixIi4pRfWc0pE3NkbNXVz+5dFxBe7sdxOEfFSRDQ1oq7e2H5EZET8XV/W%0AJTWCAb8Bi4jHI+Ll8h/eMxFxZURs1t919bWIGBERP42IZyPihYj4U0Sc0ohtZebVmXlEI9ZdKyI+%0AHBEPRsTy8rW9MSKGlvddGRFf6cK61vkykZmnZ+YFdSz7eES8s2a5JzJzs8xc3ZXH08H6vx8R/1Zz%0Ae3BErGhn2gGttx8Rt0XER3qw/fMi4rya2+dExGPlZ2pRRFzT3XXXrHN8RNzWzn0jyy8kL9Vc5vbC%0ANs+LiKt6uh71LwNe78nMzYC9gWbgC/1cT0NFxKA2Jv8IWAjsDGwNnAQ804vr71MR8Q7gq8AJmTkU%0A2A3ocdAMUHcAh9bcbgaeAA5pNQ1gdiMLiYiTKd477yw/U83ArY3cZo0tyi8um2Xm2D7aZrsGwudA%0ABrxKmfkkcBOwB0BEfCgiHihbgI9GxGkt80bENhFxfUQ8HxF/iYjfRMQbyvs+FxFPlss9FBGHl9Pf%0AEBFnRcQjEbEsIqZHxFblfS2tkJMj4omyJf35mu1tEhFTI+K5sqbPRsSimvu3L1vgS8vW05k1950X%0AETMi4qqIeBE4pY2Hvy9wZWauyMxVmfmHzLypXH587bbKaWtapW2s/5yyV2Srmvn3Kh/T4NrWcET8%0AW0R8o9W6fx4RnyqvtzxfyyNifkS8r86Xc1/g7sz8A0Bm/iUzp2bm8og4FTgR+GzZ2vtFR9uKiN2A%0Ay4ADy/mfL6ev6QVo7/0QET8CdgJ+US772ZrXelC57FYR8e8R8VT5+l7X0TrbeKx3ALtFxDbl7UOA%0AacCmrabdnZkra7cfEReW9323rO+7Net9Z0QsKLf/vYiIOp/3mzPzkfJ5fzozL2+5MyI2j4gfRMTi%0A8jPylSh3FZTvhZ/WzPsvEXFrndttV0T8r/Iz81xE3BwRO9fc962IWBgRL0bE7Ig4pJx+FHAOMClq%0AegSiVW9M1LTya57XD0fEE8B/ldMPiIi7yudxbkSM78njURdlppcN9AI8TtHaANgRmAdcUN4+GtgF%0ACOAdwF+Bvcv7vkbxT39weTmknO+tFC3h7cv5RgK7lNenAL8DRgAbA98HflIzXwJXAJsAY4FXgd3K%0A+y8Cbge2LJe/H1hU3vcGipbZl4CNgDcDjwJHlvefB6wE3lvOu0kbz8OvgN8Ck4GdWt03vmVb7Txv%0A66yf4p/bR2vmvxi4rLx+CnBnef3Q8vmK8vaWwMs1z99xwPbleicBK4DhrdfTxuM5pFzP+cDBwMat%0A7r8S+EqraV3aVu062ns/tH6uWr3Wg8rbN1D0LmxZLvuOztbZxuN9DHhfef164DDg6lbTvtTO9m8D%0APtJqfVkuswXFF5SlwFF1fJ4+CPwF+AxF672p1f0/o3jfbwpsC9wLnFbe90bgv8vn+hDgWWBEHdtc%0A6/G0uu8Y4GGKHpxBFL1zd7Wqd+vyvk8DTwNDat7XV7X3vm89T00d/698fJsAOwDLgHdTvK/+vrw9%0ArL//920oF1vwuq5sld1JEaJfBcjMGzLzkSzcDtzC692eK4HhwM6ZuTIzf5PFp3w1RXjvHhGDM/Px%0ALFszwOnA5zNzUWa+SvHP4dhYuyvv/Mx8OTPnAnMpgh7geOCrmflcZi4Cvl2zzL4U/zC+nJl/y8xH%0AKb4oTK6Z5+7MvC4zX8vMl9t4Do4DfgN8EXgsIuZExL5deA5br//HwAkAZQtscjmttd9Q/FNseV6P%0ALdf1FEBmXpuZT5XrvQZYAOzXWTGZ+Rvg/RS7XW4AlkXEN6ODgWXd3VapvfdDhyJiOPAu4PTytV1Z%0Avte6us7bgUPLFv5+FF8kf1Mz7eBynq64KDOfz8wngF8D4zpbIDOvAs4Ajiy3tyQiPlc+1u0ogu6f%0AsugpWgL8H8r3aWb+laJ7/5vAVcAZ5Xu9Xs+WreTnI+Kfy2mnA1/LzAcycxXFZ3tcSys+M6/KzGVZ%0A9FpdQvHZfWsXttmW88rH9zLFF4gbM/PG8n31S2BW+TyoDxjwem9mbpGZO2fmx1sCMCLeFRG/K7tH%0An6f4ULZ0eV5M0TK4JYru+7MAMvNh4J8owntJREyLiO3LZXYGftbyTwh4gOILwXY1tTxdc/2vQMuA%0Av+0pWrotaq/vDGxf88/teYruxe3amX8dZbiclZmjy+XmUHzxqbd7tPX6f0rRpT2copX+GkXgtN5u%0AUnQnn1BO+gBFyxOAiPjH8stGy+Pag9dfgw5l5k2Z+R5gK4qW3ClAu4PJerIt2nk/1GFH4C+Z+VwP%0A19myH35P4NEyLO+smbYJcE+dNbVo773YoSwGUb6TovV/OnBBRBxJ8T4dDCyueY6/T9GSb1n2Hore%0ApwCmd7HebcrP8RaZ2bLbZ2fgWzXb+0u57h0AIuKfy+77F8r7N6f+17w9rT+bx7X6bL6d4oub+oAB%0Ar3VExMYUIfUNYLvM3AK4keKfA5m5PDM/nZlvBiYCn4pyX3tm/jgz307x4U7gX8rVLgTeVfNPaIvM%0AHJLFvv/OLKbomm+xY831hcBjrdY7NDNrWwl1nzIxM58tH/f2FOG4gqL7FICyFTys9WKt1vEcRY/H%0AJIrQntZB6/MnFD0ZOwP7UzzvlLevAD4JbF2+Bn+ifA268Hhey8xbKXYb7NFWvXVsq8Pnr6P3QyfL%0ALgS2iogturjO1u6g6O05mte/SM2jeJ8cDfw+M19pr/yOHlt3lb0O11LsTtqD4rG+ytpB/KbySyUA%0AEfEJilb0U8Bne6GMhRS7AGo/G5tk5l3l/vbPUvSObVm+5i/Q8Wu+1mcB+B9tzFO73ELgR622v2lm%0AXtTjR6a6GPBqy0YU/2iWAqsi4l3Amp92RcSEiPi7soX7AkVL/LWIeGtEHFZ+QXiFYj/wa+VilwEX%0AtnQPRsSwiDimznqmA2dHxJYRsQNFELW4F1gexeC+TSKiKSL26EoXezmgaY8oBl4NBT4GPJyZyyj2%0Aiw6JiKMjYjDFfsyN61jtj4F/pOh2b6t7HoAsBsI9C/xfigFaz5d3bUrxz3JpWeOHeD2gO3s8x0TE%0A5PL5iojYj2Icxe/KWZ6hGKvQorNtPQOMiIiN2tlem++HdrZV+9gXUwzsvLSsdXBEHFrHOluv5+Fy%0AO1MoA778QnVPOe2OtpbrrL6uimIA5dERMTSKQYbvAkYD95SP9Rbgkoh4U3n/LlH84oGIeAvwFYpu%0A7ZMoBkF2ulugE5dRfG5Gl9vYPCKOK+8bCqyieM0HRcSXgDfVLPsMMDLWHtg4B5hcvk7NFO/tjlwF%0AvCcijiw/l0OiGLQ6opPl1EsMeK0jM5cDZ1IE63MUrdCZNbPsSjEw7SXgbuDSzPw1RfBdRBFYT1N0%0AP55dLvOtch23RMRyirDZv86SvgwsohhM9StgBkVriCx+zzyBYh/pY7welpt34SG/kWIA1PMUXaQ7%0AU7QaycwXgI+X63ySohVTz77RmRTP09NZjCnoyI+Bd1LzRSAz5wOXUDy/z1B0Nf+2zsfzHPBRiv3o%0AL1L8o704M1u6/39AMU7i+Yi4ro5t/RdFi/jpiHi2je21936AYrDcF1rtG651EsX+9geBJRS7eDpb%0AZ1vuoOhZqa37NxTvwY4C/lsUPSjPRcS3O5ivHi9S7B56guK99HXgY5nZcgyBf6T48jyf4jWaAQyP%0AYhzKVcC/ZObczFxQrudH5ZflbsnMn1H0oE2L4hcef6IY8wBwM/CfFF9g/0zxhby2e/3a8u+yiLiv%0AvP5FioG3z1EM4Gz3i2u5/YUUu4fOofgisZBiAKK500daRrpK642I+BgwOTPf0d+1SNJA5TcpDXgR%0AMTwiDi67Nd9K8ZOen/V3XZI0kHm0Ia0PNqIYcTyKoutzGnBpv1YkSQOcXfSSJFWQXfSSJFWQAS9J%0AUgWt1/vgt9lmmxw5cmR/lyFJUp+ZPXv2s5nZ+oBb61ivA37kyJHMmjWrv8uQJKnPRMSf65nPLnpJ%0AkirIgJckqYIMeEmSKmi93gcvSerYypUrWbRoEa+80t4J9TRQDRkyhBEjRjB48OBuLW/AS1KFLVq0%0AiKFDhzJy5EiKk/NpfZCZLFu2jEWLFjFq1KhurcMuekmqsFdeeYWtt97acF/PRARbb711j3peDHhJ%0AqjjDff3U09fNgJckNVRTUxPjxo1j9OjRjB07lksuuYTXXnsNgFmzZnHmmWe2udzIkSN59tlne7z9%0A6667jvnz5/d4PV3x7ne/m+eff75Pt9ma++AlaUPS2635Ok5YtskmmzBnzhwAlixZwgc+8AFefPFF%0Azj//fJqbm2lubu7dmlq57rrrmDBhArvvvnuvrnf16tU0NTW1ed+NN97Yq9vqDlvwkqQ+s+2223L5%0A5Zfz3e9+l8zktttuY8KECQAsW7aMI444gtGjR/ORj3yE9s52utlmm/H5z3+esWPHcsABB/DMM88A%0A8Pjjj3PYYYcxZswYDj/8cJ544gnuuusuZs6cyWc+8xnGjRvHI488sta6rr32WvbYYw/Gjh3LoYce%0ACsCVV17JJz/5yTXzTJgwgdtuu23Ntj/96U8zduxYvva1r3Hcccetma/2sbT0Ppx11ll873vfWzPP%0Aeeedxze+8Q0ALr74Yvbdd1/GjBnDueee25OntU0GvCSpT735zW9m9erVLFmyZK3p559/Pm9/+9uZ%0AN28e73vf+3jiiSfaXH7FihUccMABzJ07l0MPPZQrrrgCgDPOOIOTTz6Z+++/nxNPPJEzzzyTgw46%0AiIkTJ3LxxRczZ84cdtlll7XW9eUvf5mbb76ZuXPnMnPmzE5rX7FiBfvvvz9z587lrLPO4p577mHF%0AihUAXHPNNUyePHmt+SdNmsT06dPX3J4+fTqTJk3illtuYcGCBdx7773MmTOH2bNnc8cdd3T+5HWB%0AAS9JGhDuuOMOPvjBDwJw9NFHs+WWW7Y530YbbbSmpbzPPvvw+OOPA3D33XfzgQ98AICTTjqJO++8%0As9NtHnzwwZxyyilcccUVrF69utP5m5qa+Id/+AcABg0axFFHHcUvfvELVq1axQ033MAxxxyz1vx7%0A7bUXS5Ys4amnnmLu3LlsueWW7Ljjjtxyyy3ccsst7LXXXuy99948+OCDLFiwoNPtd4X74CVJferR%0ARx+lqamJbbfdlgceeKDLyw8ePHjNCPOmpiZWrVrV7Vouu+wy7rnnHm644Qb22WcfZs+ezaBBg9YM%0AAgTW+qnakCFD1trvPnnyZL773e+y1VZb0dzczNChQ9fZxnHHHceMGTN4+umnmTRpElD8zv3ss8/m%0AtNNO63btnbEF31qEl764SNogLV26lNNPP51PfvKT6/wM7NBDD+XHP/4xADfddBPPPfdcl9Z90EEH%0AMW3aNACuvvpqDjnkEACGDh3K8uXL21zmkUceYf/99+fLX/4yw4YNY+HChYwcOZI5c+bw2muvsXDh%0AQu699952t/mOd7yD++67jyuuuGKd7vkWkyZNYtq0acyYMWPNPvsjjzySH/7wh7z00ksAPPnkk+vs%0AsugpW/CSpIZ6+eWXGTduHCtXrmTQoEGcdNJJfOpTn1pnvnPPPZcTTjiB0aNHc9BBB7HTTjt1aTvf%0A+c53+NCHPsTFF1/MsGHD+Pd//3egaGV/9KMf5dvf/jYzZsxYaz/8Zz7zGRYsWEBmcvjhhzN27FgA%0ARo0axe67785uu+3G3nvv3e42m5qamDBhAldeeSVTp05tc57Ro0ezfPlydthhB4YPHw7AEUccwQMP%0APMCBBx4IFIP3rrrqKrbddtsuPeaORHujFNcHzc3N2evng7d12TfW4/edtD554IEH2G233fq7DHVT%0AW69fRMzOzE5/W2gXvSRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCSpoS688EJGjx7N%0AmDFjGDduHPfcc0+P1zlz5kwuuuiiXqiu+A16FXmgG0nagMT5vXusjzy342Na3H333Vx//fXcd999%0AbLzxxjz77LP87W9/q2vdq1atYtCgtmNq4sSJTJw4scv1bkhswUuSGmbx4sVss802bLzxxgBss802%0AbL/99mtOpwowa9Ysxo8fDxSnUz3ppJM4+OCDOemkkzjggAOYN2/emvWNHz+eWbNmrTml6wsvvMDO%0AO++85tjxK1asYMcdd2TlypU88sgjHHXUUeyzzz4ccsghPPjggwA89thjHHjggey555584Qtf6MNn%0Ao28Z8JKkhjniiCNYuHAhb3nLW/j4xz/O7bff3uky8+fP51e/+hU/+clP1jrd6uLFi1m8eDHNza8f%0AxG3zzTdn3Lhxa9Z7/fXXc+SRRzJ48GBOPfVUvvOd7zB79my+8Y1v8PGPfxyAKVOm8LGPfYw//vGP%0Aaw4dW0UGvCSpYTbbbDNmz57N5ZdfzrBhw5g0aRJXXnllh8tMnDiRTTbZBIDjjz+eGTNmAMW51I89%0A9th15p80aRLXXHMNANOmTWPSpEm89NJL3HXXXRx33HGMGzeO0047jcWLFwPw29/+lhNOOAEoTitb%0AVe6DlyQ1VFNTE+PHj2f8+PHsueeeTJ06da1TstaejhVg0003XXN9hx12YOutt+b+++/nmmuu4bLL%0ALltn/RMnTuScc87hL3/5C7Nnz+awww5jxYoVbLHFFsyZM6fNmlqfya6KbMFLkhrmoYceYsGCBWtu%0Az5kzh5133pmRI0cye/ZsAH760592uI5Jkybx9a9/nRdeeIExY8asc/9mm23Gvvvuy5QpU5gwYQJN%0ATU286U1vYtSoUVx77bVAcf71uXPnAnDwwQevdVrZqjLgJUkN89JLL3HyySez++67M2bMGObPn895%0A553Hueeey5QpU2hubqapqanDdRx77LFMmzaN448/vt15Jk2axFVXXcWkSZPWTLv66qv5wQ9+wNix%0AYxk9ejQ///nPAfjWt77F9773Pfbcc0+efPLJ3nmgA5Cni21tA+i2GRDW4/edtD7xdLHrN08XK0mS%0A1mLAS5JUQQa8JEkVZMBLUsWtz2OtNmQ9fd0MeEmqsCFDhrBs2TJDfj2TmSxbtowhQ4Z0ex0e6EaS%0AKmzEiBEsWrSIpUuX9ncp6qIhQ4YwYsSIbi9vwEtShQ0ePJhRo0b1dxnqB3bRS5JUQQa8JEkVZMBL%0AklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JU%0AQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEND/iIaIqIP0TE9eXtrSLilxGxoPy7Zc28Z0fEwxHxUEQc%0A2ejaJEmqqr5owU8BHqi5fRZwa2buCtxa3iYidgcmA6OBo4BLI6KpD+qTJKlyGhrwETECOBr4vzWT%0AjwGmltenAu+tmT4tM1/NzMeAh4H9GlmfJElV1egW/L8CnwVeq5m2XWYuLq8/DWxXXt8BWFgz36Jy%0A2loi4tSImBURs5YuXdqAkiVJWv81LOAjYgKwJDNntzdPZiaQXVlvZl6emc2Z2Txs2LCelilJUiUN%0AauC6DwYmRsS7gSHAmyLiKuCZiBiemYsjYjiwpJz/SWDHmuVHlNMkSVIXNawFn5lnZ+aIzBxJMXju%0AvzLzg8BM4ORytpOBn5fXZwKTI2LjiBgF7Arc26j6JEmqska24NtzETA9Ij4M/Bk4HiAz50XEdGA+%0AsAr4RGau7of6JEla70WxG3z91NzcnLNmzerdlUb07vrUtvX4fSdJ/SkiZmdmc2fzeSQ7SZIqyICX%0AJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySp%0Aggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIM%0AeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJ%0AkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpgjoN+IjY%0ANCLeUF5/S0RMjIjBjS9NkiR1Vz0t+DuAIRGxA3ALcBJwZSOLkiRJPVNPwEdm/hV4P3BpZh4HjG5s%0AWZIkqSfqCviIOBA4EbihnNbUuJIkSVJP1RPwU4CzgZ9l5ryIeDPw68aWJUmSemJQR3dGRBMwMTMn%0AtkzLzEeBMxtdmCRJ6r4OW/CZuRp4ex/VIkmSekmHLfjSHyJiJnAtsKJlYmb+R8OqkiRJPVJPwA8B%0A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%0ASZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmS%0AKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirI%0AgJckqYIMeEmSKsiAlySpggx4SZIqaFB/F6ANU5wf/V3CBiHPzf4uQVI/sQUvSVIFGfCSJFVQwwI+%0AIn4YEUsi4k8107aKiF9GxILy75Y1950dEQ9HxEMRcWSj6pIkaUPQyBb8lcBRraadBdyambsCt5a3%0AiYjdgcnA6HKZSyOiqYG1SZJUaQ0L+My8A/hLq8nHAFPL61OB99ZMn5aZr2bmY8DDwH6Nqk2SpKrr%0A633w22Xm4vL608B25fUdgIU18y0qp60jIk6NiFkRMWvp0qWNq1SSpPVYvw2yy8wEuvwbnsy8PDOb%0AM7N52LBhDahMkqT1X18H/DMRMRyg/LuknP4ksGPNfCPKaZIkqRv6OuBnAieX108Gfl4zfXJEbBwR%0Ao4BdgXv7uDZJkiqjYUeyi4ifAOOBbSJiEXAucBEwPSI+DPwZOB4gM+dFxHRgPrAK+ERmrm5UbZIk%0AVV3DAj4zT2jnrsPbmf9C4MJG1SNJ0obEI9lJklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkV%0AZMBLklRBDfsdvCR1S0R/V7BhOK+/C1Cj2YKXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIq%0AyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiA%0AlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJck%0AqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmC%0ADHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4%0ASZIqyICXJKmCBlzAR8RREfFQRDwcEWf1dz2SJK2PBlTAR0QT8D3gXcDuwAkRsXv/ViVJ0vpnQAU8%0AsB/wcGY+mpl/A6YBx/RzTZIkrXcGWsDvACysub2onCZJkrpgUH8X0FURcSpwannzpYh4qD/rUTed%0A198FdMs2wLP9XURXxHnR3yVooDqvvwvoFj+DhZ3rmWmgBfyTwI41t0eU09bIzMuBy/uyKAkgImZl%0AZnN/1yFtqPwMds1A66L/PbBrRIyKiI2AycDMfq5JkqT1zoBqwWfmqoj4JHAz0AT8MDPn9XNZkiSt%0AdwZUwANk5o3Ajf1dh9QGdw1J/cvPYBdEZvZ3DZIkqZcNtH3wkiSpFxjwUjdExPiIuL6/65DWJxFx%0AZkQ8EBFXN2j950XEPzdi3eujAbcPXpJUWR8H3pmZi/q7kA2BLXhtsCJiZEQ8GBFXRsR/R8TVEfHO%0AiPhtRCyIiP3Ky90R8YeIuCsi3trGejaNiB9GxL3lfB5eWWolIi4D3gzcFBGfb+szExGnRMR1EfHL%0AiHg8Ij4ZEZ8q5/ldRGxVzvfRiPh9RMyNiJ9GxBvb2N4uEfGfETE7In4TEW/r20fc/wx4bej+DrgE%0AeFt5+QDwduCfgXOAB4FDMnMv4EvAV9tYx+eB/8rM/YD/CVwcEZv2Qe3SeiMzTweeoviMbEr7n5k9%0AgPcD+wIXAn8tP393A/9YzvMfmblvZo4FHgA+3MYmLwfOyMx9KD7PlzbmkQ1cdtFrQ/dYZv4RICLm%0AAbdmZkbEH4GRwObA1IjYFUhgcBvrOAKYWLPvbwiwE8U/Hknrau8zA/DrzFwOLI+IF4BflNP/CIwp%0Ar+8REV8BtgA2ozh2yhoRsRlwEHBtxJpDxW7ciAcykBnw2tC9WnP9tZrbr1F8Pi6g+IfzvogYCdzW%0AxjoC+IfM9LwIUn3a/MxExP50/pkEuBJ4b2bOjYhTgPGt1v8G4PnMHNe7Za9f7KKXOrY5r58P4ZR2%0A5rkZOCPKpkJE7NUHdUnrs55+ZoYCiyNiMHBi6zsz80XgsYg4rlx/RMTYHta83jHgpY59HfhaRPyB%0A9nu8LqDour+/7Oa/oK+Kk9ZTPf3MfBG4B/gtxTiZtpwIfDgi5gLzgA1u8KtHspMkqYJswUuSVEEG%0AvCRJFWTAS5JUQQa8JEkVZMBLklRBBrykNpXHC58XEfdHxJzyICSS1hMeyU7SOiLiQGACsHdmvhoR%0A2wAb9XNZkrrAFryktgwHns3MVwEy89nMfCoi9omI28szdN0cEcMjYlB5Zq/xABHxtYi4sD+Ll+SB%0AbiS1oTxZx53AG4FfAdcAdwG3A8dk5tKImAQcmZn/KyJGAzOAM4CLgf0z82/9U70ksIteUhsy86WI%0A2Ac4hOJ0ntcAX6E4lecvy0OINwGLy/nnRcSPgOuBAw13qf8Z8JLalJmrKc6ed1t5+txPAPMy88B2%0AFtkTeB7Ytm8qlNQR98FLWkdEvDUidq2ZNI7i/PbDygF4RMTgsmueiHg/sBVwKPCdiNiir2uWtDb3%0AwUtaR9k9/x1gC2AV8DBwKjAC+DbFaXQHAf8K/Ixi//zhmbkwIs4E9snMk/ujdkkFA16SpAqyi16S%0ApAoy4CVJqiADXpKkCjLgJUmqIANekqQKMuAlSaogA16SpAoy4CVJqqD/D4fPhy9k8RkrAAAAAElF%0ATkSuQmCC%0A)

Examining the survival statistics, a large majority of males did not
survive the ship sinking. However, a majority of females *did* survive
the ship sinking. Let's build on our previous prediction: If a passenger
was female, then we will predict that they survived. Otherwise, we will
predict the passenger did not survive.\
 Fill in the missing code below so that the function will make this
prediction.\
 **Hint:** You can access the values of each feature for a passenger
like a dictionary. For example, `passenger['Sex']` is the sex of the
passenger.

In [144]:

    def predictions_1(data):
        """ Model with one feature: 
                - Predict a passenger survived if they are female. """
        
        predictions = []
        for _, passenger in data.iterrows():
            
            # Remove the 'pass' statement below 
            # and write your prediction conditions here
            if (passenger['Sex'] == 'female'): 
                predictions.append(1)
            else:
                predictions.append(0)
                
        
        # Return our predictions
        return pd.Series(predictions)

    # Make the predictions
    predictions = predictions_1(data)

### Question 2[¶](#Question-2) {#Question-2}

*How accurate would a prediction be that all female passengers survived
and the remaining passengers did not survive?*\
 **Hint:** Run the code cell below to see the accuracy of this
prediction.

In [145]:

    print accuracy_score(outcomes, predictions)

    Predictions have an accuracy of 78.68%.

**Answer**: Predictions that all female passengers survived and the
remaining passengers did not survive, have an accuracy of 78.68%.

* * * * *

Using just the **Sex** feature for each passenger, we are able to
increase the accuracy of our predictions by a significant margin. Now,
let's consider using an additional feature to see if we can further
improve our predictions. For example, consider all of the male
passengers aboard the RMS Titanic: Can we find a subset of those
passengers that had a higher rate of survival? Let's start by looking at
the **Age** of each male, by again using the `survival_stats` function.
This time, we'll use a fourth parameter to filter out the data so that
only passengers with the **Sex** 'male' will be included.\
 Run the code cell below to plot the survival outcomes of male
passengers based on their age.

In [146]:

    vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

![](data:image/png;base64,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%0A9OnTed/73veufX3jG9/g9ttvZ8aMGUyePHmFsS9ZsoQ99tiDGTNmcMopp3D//fezZMkSAK655hrG%0AjBnzrvVHjx7Ntddeu2z62muvZfTo0UyZMoXZs2fzwAMPMH36dKZNm8Y999yz4idvFZnsJUmrhXvu%0AuYdPfepTABx22GFsvPHGba639tprL6tB77bbbjz99NMA3HfffXzyk58EYOzYsfz2t79d4TH32Wcf%0AjjvuOC699FLeeuutFa7fq1cv/vmf/xmA3r17c+ihh/LLX/6SN998k1tuuYXDDz/8XevvsssuLFiw%0AgL/+9a/MmDGDjTfemK233popU6YwZcoUdtllF3bddVceffRRZs+evcLjryrP2UuSutSTTz5Jr169%0A2HzzzZk1a9ZKb9+nT59lPdV79erFm2++ucqxXHTRRdx///3ccsst7LbbbkybNo3evXsv60AIvOvy%0At759+77rPP2YMWP44Q9/yCabbMKIESPYYIMNljvGUUcdxfXXX8+zzz7L6NGjgeI6+lNPPZXxje5D%0A0Q6TvbpPF73Ju00dHZekNc3ChQs58cQT+dKXvrTcpWX7778/V111Faeffjq33XYbL7744krte++9%0A92bSpEmMHTuWK6+8kv322w+ADTbYgMWLF7e5zRNPPMEee+zBHnvswW233cacOXMYNGgQF154IW+/%0A/Tbz5s3jgQceaPeYH/rQh/j0pz/NpZdeulwTfovRo0fzuc99jueff567774bgEMOOYQzzjiDY489%0Aln79+jFv3jz69OnD5ptvvlJlrpfJXpLUVK+99hrDhw9n6dKl9O7dm7Fjx/KVr3xlufXOPPNMjjnm%0AGIYMGcLee+/NNttss1LH+cEPfsDxxx/PeeedR//+/fnZz34GFLXvz33uc3z/+9/n+uuvf9d5+69+%0A9avMnj2bzOSggw5i2LBhAAwePJgdd9yRHXbYgV133bXdY/bq1YuRI0dy+eWXM3HixDbXGTJkCIsX%0AL2arrbZiwIABABx88MHMmjWLvfbaCyg6/l1xxRVNS/bRXm/HnmDEiBHpePY9mDV7qelmzZrFDjvs%0A0N1haBW19fpFxLTMXKnrFe2gJ0lSxTUt2UfETyNiQUQ8XDPvvIh4NCIeiohfRMRGNctOjYjHI+Kx%0AiDikWXFJkrSmaWbN/nLg0Fbz7gB2ysyhwP8CpwJExI7AGGBIuc2FEdH2bYkkSdJKaVqyz8x7gBda%0A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%0AbJSzDrwUMlwAAAAASUVORK5CYII=%0A)

Examining the survival statistics, the majority of males younger than 10
survived the ship sinking, whereas most males age 10 or older *did not
survive* the ship sinking. Let's continue to build on our previous
prediction: If a passenger was female, then we will predict they
survive. If a passenger was male and younger than 10, then we will also
predict they survive. Otherwise, we will predict they do not survive.\
 Fill in the missing code below so that the function will make this
prediction.\
 **Hint:** You can start your implementation of this function using the
prediction code you wrote earlier from `predictions_1`.

In [147]:

    def predictions_2(data):
        """ Model with two features: 
                - Predict a passenger survived if they are female.
                - Predict a passenger survived if they are male and younger than 10. """
        
        predictions = []
        for _, passenger in data.iterrows():
            
            # Remove the 'pass' statement below 
            # and write your prediction conditions here
            if (passenger['Sex'] == 'female'):
                predictions.append(1)
            elif (passenger['Sex'] == 'male' and passenger['Age'] < 10): 
                predictions.append(1)
            else:
                predictions.append(0)
        
        # Return our predictions
        return pd.Series(predictions)

    # Make the predictions
    predictions = predictions_2(data)

### Question 3[¶](#Question-3) {#Question-3}

*How accurate would a prediction be that all female passengers and all
male passengers younger than 10 survived?*\
 **Hint:** Run the code cell below to see the accuracy of this
prediction.

In [148]:

    print accuracy_score(outcomes, predictions)

    Predictions have an accuracy of 79.35%.

**Answer**: Assuming that all female passengers and all male passengers
younger than 10 survived, our predictions have an accuracy of 79.35%.

* * * * *

Adding the feature **Age** as a condition in conjunction with **Sex**
improves the accuracy by a small margin more than with simply using the
feature **Sex** alone. Now it's your turn: Find a series of features and
conditions to split the data on to obtain an outcome prediction accuracy
of at least 80%. This may require multiple features and multiple levels
of conditional statements to succeed. You can use the same feature
multiple times with different conditions.\
 **Pclass**, **Sex**, **Age**, **SibSp**, and **Parch** are some
suggested features to try.

Use the `survival_stats` function below to to examine various survival
statistics.\
 **Hint:** To use mulitple filter conditions, put each condition in the
list passed as the last argument. Example:
`["Sex == 'male'", "Age < 18"]`

In [149]:

    vs.survival_stats(data, outcomes, 'Embarked', ["Sex == 'female'", "Pclass < 3"])

![](data:image/png;base64,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%0APn06Y8aMAWDZsmUcddRRDB8+nI985CO0dWW/QYMG8YUvfIGRI0ey//7788wzzwDw+OOPc/jhhzNi%0AxAiOOOIInnzySX7/+98zbdo0PvOZzzBq1CgeeeSR183rxhtvZI899mDkyJEccsghAEyaNImzzjpr%0A9Thjxoxh+vTpq9v+9Kc/zciRI/na177GCSecsHq82ufSvPVhwoQJfP/73189zvnnn883v/lNAC6+%0A+GL23XdfRowYwcSJEzvzskqS1sKA7wZvfOMbWbVqFYsXL37d8AsuuIC3v/3tzJkzh/e85z08+eST%0ArU6/YsUK9t9/f2bPns0hhxzClVdeCcAnP/lJTj31VB544AFOOeUUzj77bA488EDGjh3LxRdfzKxZ%0As9hll11eN68vfelL/PznP2f27NlMmzZtrbWvWLGC/fbbj9mzZzNhwgTuvfdeVqxYAcD111/PSSed%0A9Lrxx40bxw033LD6/g033MC4ceO44447mDdvHvfddx+zZs1i5syZ3H333Wt/8SRJHWLA96C7776b%0AD3zgAwAce+yxbLHFFq2Ot+GGG67uKe+zzz48/vjjANxzzz28//3vB2D8+PH89re/XWubBx10EB/6%0A0Ie48sorWbVq1VrHb2pq4n3vex8A/fr145hjjuFnP/sZr776KrfeeivHHXfc68bfa6+9WLx4MU89%0A9RSzZ89miy22YMcdd+SOO+7gjjvuYK+99mLvvffmoYceYt68eWttX5LUMdXfB98LPProozQ1NbHN%0ANtvw4IMPrvP0/fv3X32EeVNTE6+++mqHa7n88su59957ufXWW9lnn32YOXMm/fr1W30QIPC6n6oN%0AGDDgdfvdTzrpJC699FK23HJLRo8ezSabbLJGGyeccAJTp07l6aefZty4cUDxO/fPfe5znHHGGR2u%0AXZJUPwO+wZYsWcKZZ57JWWedtcbPwA455BCuvfZazj33XG6//XaeffbZdZr3gQceyJQpUxg/fjzX%0AXHMNBx98MACbbLIJy5cvb3WaRx55hP3224/99tuP22+/nfnz5zN06FAuu+wyXnvtNRYuXMh9993X%0AZpuHHnoo//qv/8qVV165xub5ZuPGjeOjH/0oS5cu5a677gLg6KOP5rzzzuOUU05h0KBBLFy4kP79%0A+7PNNtus03OWVA1xwfrxs9icuPYDsxvFgG+Al156iVGjRrFy5Ur69evH+PHj+dSnPrXGeBMnTuTk%0Ak09m+PDhHHjggey0007r1M73vvc9TjvtNC6++GIGDx7Mj370I6DoZX/0ox/lu9/9LlOnTn3dfvjP%0AfOYzzJs3j8zkiCOOYOTIkQAMGzaM3Xffnd1224299967zTabmpoYM2YMkyZNYvLkya2OM3z4cJYv%0AX84OO+zAdtttB8BRRx3Fgw8+yAEHHAAUB+9dffXVBrwkNUi0deR2XzB69OhseT34Bx98kN12262H%0AKlJn+f5J6wd78B0XETMzc62/t/YgO0mSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIM+Aa5%0A8MILGT58OCNGjGDUqFHce++9nZ7ntGnTuOiii7qguuJ36JKk6qr8iW66+reW9fym8Z577uGWW27h%0A/vvvZ6ONNmLp0qX8/e9/r2v+r776Kv36tf62jB07lrFjx65TvZKk9ZM9+AZYtGgRW2+9NRtttBEA%0AW2+9Ndtvv/3qS6oCzJgxg8MOOwwoLqk6fvx4DjroIMaPH8/+++/PnDlzVs/vsMMOY8aMGasv6/r8%0A88+z8847rz5//IoVK9hxxx1ZuXIljzzyCMcccwz77LMPBx98MA899BAAjz32GAcccAB77rkn5557%0Abje+GpKknmDAN8BRRx3F/PnzefOb38zHP/7x1edjb8/cuXP55S9/yXXXXfe6S64uWrSIRYsWMXr0%0AP05atNlmmzFq1KjV873llls4+uij6d+/P6effjrf+973mDlzJt/85jf5+Mc/DsA555zDxz72Mf70%0Apz+tPn2sJKm6DPgGGDRoEDNnzuSKK65g8ODBjBs3jkmTJrU7zdixYxk4cCAAJ554IlOnTgWK66kf%0Af/zxa4w/btw4rr/+egCmTJnCuHHjePHFF/n973/PCSecwKhRozjjjDNYtGgRAL/73e84+eSTgeLS%0AspKkaqv8Pvie0tTUxGGHHcZhhx3GnnvuyeTJk193WdbaS7ICbLzxxqtv77DDDmy11VY88MADXH/9%0A9Vx++eVrzH/s2LF8/vOf569//SszZ87k8MMPZ8WKFWy++ebMmjWr1ZpaXs1OklRd9uAb4OGHH2be%0AvHmr78+aNYudd96ZoUOHMnPmTAB+/OMftzuPcePG8Y1vfIPnn3+eESNGrPH4oEGD2HfffTnnnHMY%0AM2YMTU1NbLrppgwbNowbb7wRKK7BPnv2bAAOOuggpkyZAsA111zTJc9TktR7GfAN8OKLL3Lqqaey%0A++67M2LECObOncv555/PxIkTOeeccxg9ejRNTU3tzuP4449nypQpnHjiiW2OM27cOK6++mrGjRu3%0Aetg111zDVVddxciRIxk+fDg333wzAN/5znf4/ve/z5577snChQu75olKknqthl0uNiL+HzAGWJyZ%0Ae5TDtgSuB4YCjwMnZuaz5WOfAz4MrALOzsyfr60NLxdbPb5/0vrBy8V2XG+4XOwk4JgWwyYAd2bm%0ArsCd5X0iYnfgJGB4Oc1lEdF+F1eSJLWpYQGfmXcDf20x+Dhgcnl7MvDumuFTMvOVzHwM+AvwtkbV%0AJklS1XX3PvhtM3NReftpYNvy9g7A/JrxFpTD1hARp0fEjIiYsWTJksZVKklSH9ZjB9llsfN/nXdO%0AZOYVmTk6M0cPHjy4rXE6W556gO+bJHWd7g74ZyJiO4Dy/+Jy+EJgx5rxhpTD1tmAAQNYtmyZYdHH%0AZCbLli1jwIABPV2KJFVCd5/oZhpwKnBR+f/mmuHXRsS3gO2BXYH7OtLAkCFDWLBgAW6+73sGDBjA%0AkCFDeroMSaqEhgV8RFwHHAZsHRELgIkUwX5DRHwYeAI4ESAz50TEDcBc4FXgE5m5qiPt9u/fn2HD%0AhnXBM5Akqe9qWMBn5sltPHREG+NfCFzYqHokSVqfeCY7SZIqyICXJKmCDHhJkirIgJckqYIMeEmS%0AKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirI%0AgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICX%0AJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySp%0Aggx4SZIqyICXJKmCDHhJkirIgJckqYIMeEmSKsiAlySpggx4SZIqyICXJKmCDHhJkirIgJckqYIM%0AeEmSKsiAlySpggx4SZIqqEcCPiL+PSLmRMSfI+K6iBgQEVtGxC8iYl75f4ueqE2SpCro9oCPiB2A%0As4HRmbkH0AScBEwA7szMXYE7y/uSJKkDemoTfT9gYET0A94APAUcB0wuH58MvLuHapMkqc/r9oDP%0AzIXAN4EngUXA85l5B7BtZi4qR3sa2La7a5MkqSp6YhP9FhS99WHA9sDGEfGB2nEyM4FsY/rTI2JG%0ARMxYsmRJw+uVJKkv6olN9EcCj2XmksxcCfwEOBB4JiK2Ayj/L25t4sy8IjNHZ+bowYMHd1vRkiT1%0AJT0R8E8C+0fEGyIigCOAB4FpwKnlOKcCN/dAbZIkVUK/7m4wM++NiKnA/cCrwB+BK4BBwA0R8WHg%0ACeDE7q5NkqSq6PaAB8jMicDEFoNfoejNS5KkTvJMdpIkVZABL0lSBRnwkiRVkAEvSVIFGfCSJFWQ%0AAS9JUgUZ8JIkVZABL0lSBRnwkiRV0FoDPiI2jogNyttvjoixEdG/8aVJkqSOqqcHfzcwICJ2AO4A%0AxgOTGlmUJEnqnHoCPjLzb8B7gcsy8wRgeGPLkiRJnVFXwEfEAcApwK3lsKbGlSRJkjqrnoA/B/gc%0A8NPMnBMRbwR+3diyJElSZ7R7udiIaALGZubY5mGZ+ShwdqMLkyRJHdduDz4zVwFv76ZaJElSF2m3%0AB1/6Y0RMA24EVjQPzMyfNKwqSZLUKfUE/ABgGXB4zbAEDHhJknqptQZ8Zp7WHYVIkqSuU8+Z7N4c%0AEXdGxJ/L+yMi4tzGlyZJkjqqnp/JXUnxM7mVAJn5AHBSI4uSJEmdU0/AvyEz72sx7NVGFCNJkrpG%0APQG/NCJ2oTiwjog4HljU0KokSVKn1HMU/SeAK4C3RsRC4DHgAw2tSpIkdUo9R9E/ChwZERsDG2Tm%0A8saXJUmSOmOtAR8Rn2pxH+B5YGZmzmpQXZIkqRPq2Qc/GjgT2KH8OwM4BrgyIj7bwNokSVIH1bMP%0Afgiwd2a+CBAREykuG3sIMBP4RuPKkyRJHVFPD34b4JWa+yuBbTPzpRbDJUlSL1FPD/4a4N6IuLm8%0A/y7g2vKgu7kNq0ySJHVYPUfRfzki/hs4sBx0ZmbOKG+f0rDKJElSh9XTgwe4H1jYPH5E7JSZTzas%0AKkmS1Cn1/Ezuk8BE4BlgFRAUZ7Ub0djSJElSR9XTgz8HeEtmLmt0MZIkqWvUcxT9fIoT20iSpD6i%0Anh78o8D0iLiVmp/FZea3GlaVJEnqlHoC/snyb8PyT5Ik9XL1/EzuAoCIeENm/q3xJUmSpM5a6z74%0AiDggIuYCD5X3R0bEZQ2vTJIkdVg9B9l9GzgaWAaQmbMpzkMvSZJ6qXoCnsyc32LQqgbUIkmSukg9%0AB9nNj4gDgYyI/hS/i3+wsWVJkqTOqKcHfybwCYprwS8ERpX3JUlSL1XPUfRL8aIykiT1KfUcRf+N%0AiNg0IvpHxJ0RsSQiPtAdxUmSpI6pZxP9UZn5AjAGeBx4E/CZRhYlSZI6p56Ab96MfyxwY2Z6XnpJ%0Aknq5eo6ivyUiHgJeAj4WEYOBlxtbliRJ6oy19uAzcwJwIDA6M1cCK4DjGl2YJEnquHoOsjsBWJmZ%0AqyLiXOBqYPvONBoRm0fE1Ih4KCIeLE+Hu2VE/CIi5pX/t+hMG5Ikrc/q2Qd/XmYuj4i3A0cCVwE/%0A6GS73wH+OzPfCoykOHHOBODOzNwVuLO8L0mSOqCegG8+Le2xwBWZeSuduGxsRGxGcS77qwAy8++Z%0A+RzFZv/J5WiTgXd3tA1JktZ39QT8woj4ITAOuC0iNqpzurYMA5YAP4qIP0bEf0bExsC2mbmoHOdp%0AYNtOtCFJ0nqtnqA+Efg5cHTZ096Szv0Ovh+wN/CDzNyL4qC9122Oz8wEsrWJI+L0iJgRETOWLFnS%0AiTIkSaqueo6i/1tm/gR4PiJ2AvpTXhu+gxYACzLz3vL+VIrAfyYitgMo/y9uo54rMnN0Zo4ePHhw%0AJ8qQJKm66jmKfmxEzAMeA+4q/9/e0QYz82mKK9S9pRx0BDAXmAacWg47Fbi5o21IkrS+q+dEN18G%0A9gd+mZl7RcQ/A509F/0ngWsiYkPgUeA0ii8bN0TEh4EnKHYNSJKkDqgn4Fdm5rKI2CAiNsjMX0fE%0AtzvTaGbOAka38tARnZmvJEkq1BPwz0XEIOBuil73YooD4yRJUi9Vz1H0xwF/A/4d+G/gEeBdjSxK%0AkiR1Trs9+Ih4N8XlYf+UmT/nHyeikSRJvVibPfiIuIyi174V8OWIOK/bqpIkSZ3SXg/+EGBkeZGZ%0ANwC/oTiiXpIk9XLt7YP/e2auguJkN0B0T0mSJKmz2uvBvzUiHihvB7BLeT8oziY7ouHVSZKkDmkv%0A4HfrtiokSVKXajPgM/OJ7ixEkiR1nc5c9lWSJPVSBrwkSRXU3u/g7yz/f737ypEkSV2hvYPstouI%0AA4GxETGFFj+Ty8z7G1qZJEnqsPYC/ovAecAQ4FstHkvg8EYVJUmSOqe9o+inAlMj4rzM9Ax2kiT1%0AIWu9XGxmfjkixlKcuhZgembe0tiyJElSZ6z1KPqI+BpwDjC3/DsnIr7a6MIkSVLHrbUHDxwLjMrM%0A1wAiYjLwR+DzjSxMkiR1XL2/g9+85vZmjShEkiR1nXp68F8D/hgRv6b4qdwhwISGViVJkjqlnoPs%0ArouI6cC+5aD/yMynG1qVJEnqlHp68GTmImBag2uRJEldxHPRS5JUQQa8JEkV1G7AR0RTRDzUXcVI%0AkqSu0W7RTrTNAAAMEklEQVTAZ+Yq4OGI2Kmb6pEkSV2gnoPstgDmRMR9wIrmgZk5tmFVSZKkTqkn%0A4M9reBXqdnFBrH2kCsiJ2dMlSFKPqOd38HdFxM7Arpn5y4h4A9DU+NIkSVJH1XOxmY8CU4EfloN2%0AAG5qZFGSJKlz6vmZ3CeAg4AXADJzHrBNI4uSJEmdU0/Av5KZf2++ExH9AHdsSpLUi9UT8HdFxOeB%0AgRHxL8CNwM8aW5YkSeqMegJ+ArAE+BNwBnAbcG4ji5IkSZ1Tz1H0r0XEZOBeik3zD2emm+glSerF%0A1hrwEXEscDnwCMX14IdFxBmZeXuji5MkSR1Tz4luLgH+OTP/AhARuwC3Aga8JEm9VD374Jc3h3vp%0AUWB5g+qRJEldoM0efES8t7w5IyJuA26g2Ad/AvCHbqhNkiR1UHub6N9Vc/sZ4NDy9hJgYMMqkiRJ%0AndZmwGfmad1ZiCRJ6jr1HEU/DPgkMLR2fC8XK0lS71XPUfQ3AVdRnL3utcaWI0mSukI9Af9yZn63%0A4ZVIkqQuU0/AfyciJgJ3AK80D8zM+xtWlSRJ6pR6An5PYDxwOP/YRJ/lfUmS1AvVE/AnAG+svWSs%0AJEnq3eo5k92fgc0bXYgkSeo69fTgNwceiog/8Pp98P5MTpKkXqqegJ/YiIYjogmYASzMzDERsSVw%0APcXv7R8HTszMZxvRtiRJVVfP9eDvalDb5wAPApuW9ycAd2bmRRExobz/Hw1qW5KkSlvrPviIWB4R%0AL5R/L0fEqoh4oTONRsQQ4FjgP2sGHwdMLm9PBt7dmTYkSVqf1dOD36T5dkQERRDv38l2vw18Ftik%0AZti2mbmovP00sG0n25Akab1Vz1H0q2XhJuDojjYYEWOAxZk5s712KH5r39r0p0fEjIiYsWTJko6W%0AIUlSpdVzsZn31tzdABgNvNyJNg8CxkbEO4EBwKYRcTXwTERsl5mLImI7YHFrE2fmFcAVAKNHj271%0AS4AkSeu7enrw76r5OxpYTrGZvkMy83OZOSQzhwInAb/KzA8A04BTy9FOBW7uaBuSJK3v6tkH313X%0Ahb8IuCEiPgw8AZzYTe1KklQ5bQZ8RHyxnekyM7/c2cYzczowvby9DDiis/OUJEnt9+BXtDJsY+DD%0AwFZApwNekiQ1RpsBn5mXNN+OiE0oTkxzGjAFuKSt6SRJUs9rdx98efrYTwGnUJx8Zm9PHytJUu/X%0A3j74i4H3Uvwkbc/MfLHbqpIkSZ3S3s/kPg1sD5wLPFVzutrlnT1VrSRJaqz29sGv01nuJElS72GI%0AS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuS%0AVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRB%0ABrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8%0AJEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJFWTAS5JUQQa8JEkVZMBLklRBBrwkSRVkwEuSVEEGvCRJ%0AFWTAS5JUQd0e8BGxY0T8OiLmRsSciDinHL5lRPwiIuaV/7fo7tokSaqKnujBvwp8OjN3B/YHPhER%0AuwMTgDszc1fgzvK+JEnqgG4P+MxclJn3l7eXAw8COwDHAZPL0SYD7+7u2iRJqooe3QcfEUOBvYB7%0AgW0zc1H50NPAtj1UliRJfV6PBXxEDAJ+DPxbZr5Q+1hmJpBtTHd6RMyIiBlLlizphkolSep7eiTg%0AI6I/Rbhfk5k/KQc/ExHblY9vByxubdrMvCIzR2fm6MGDB3dPwZIk9TE9cRR9AFcBD2bmt2oemgac%0AWt4+Fbi5u2uTJKkq+vVAmwcB44E/RcSsctjngYuAGyLiw8ATwIk9UJskSZXQ7QGfmb8Foo2Hj+jO%0AWiRJqirPZCdJUgUZ8JIkVZABL0lSBRnwkiRVkAEvSVIFGfCSJFWQAS9JUgUZ8JIkVZABL0lSBRnw%0AkiRVkAEvSVIFGfCSJFWQAS9JUgUZ8JIkVZABL0lSBRnwkiRVkAEvSVIFGfCSJFWQAS9JUgUZ8JIk%0AVZABL0lSBRnwkiRVkAEvSVIFGfCSJFWQAS9JUgUZ8JIkVZABL0lSBRnwkiRVkAEvSVIFGfCSJFWQ%0AAS9JvUnE+vGnhjPgJUmqIANekqQKMuAlSaogA16SpAoy4CVJqiADXpKkCjLgJUmqIANekqQKMuAl%0ASaogA16SpAoy4CVJqiADXpKkCjLgJUmqoH49XYCkvikuqP4VwXJi9nQJUofZg5ckqYIMeEmSKsiA%0AlySpggx4SZIqqNcFfEQcExEPR8RfImJCT9cjSVJf1KsCPiKagO8D7wB2B06OiN27uYj140+SVGm9%0AKuCBtwF/ycxHM/PvwBTguB6uSZKkPqe3BfwOwPya+wvKYZIkaR1EZu85kUNEHA8ck5kfKe+PB/bL%0AzLNqxjkdOL28+xbg4W4vVB21NbC0p4tQn+YypM6oyvKzc2YOXttIve1MdguBHWvuDymHrZaZVwBX%0AdGdR6hoRMSMzR/d0Heq7XIbUGevb8tPbNtH/Adg1IoZFxIbAScC0Hq5JkqQ+p1f14DPz1Yg4C/g5%0A0AT8v8yc08NlSZLU5/SqgAfIzNuA23q6DjWEu1bUWS5D6oz1avnpVQfZSZKkrtHb9sFLkqQuYMCr%0A4SLinyJiSkQ8EhEzI+K2iHhzT9elviEihkTEzRExLyIejYhLI2Kjnq5LfUtEfCEi5kTEAxExKyL2%0A6+maGs2AV0NFRAA/BaZn5i6ZuQ/wOWDbnq1MfUG5/PwEuCkzdwV2BQYC3+jRwtSnRMQBwBhg78wc%0AARzJ60+qVkm97iA7Vc4/Aysz8/LmAZk5uwfrUd9yOPByZv4IIDNXRcS/A09ExBcy88WeLU99xHbA%0A0sx8BSAzq3Cym7WyB69G2wOY2dNFqM8aTovlJzNfAB4H3tQTBalPugPYMSL+NyIui4hDe7qg7mDA%0AS5IqrdzSsw/Fac6XANdHxId6tKhuYMCr0eZQrFhSR8ylxfITEZsC/4TXodA6yMxVmTk9MycCZwHv%0A6+maGs2AV6P9CtiovEgQABExIiIO7sGa1HfcCbwhIj4IEBFNwCXApZn5Uo9Wpj4jIt4SEbvWDBoF%0APNFT9XQXA14NlcWZlN4DHFn+TG4O8DXg6Z6tTH1BzfJzfETMA5YBr2XmhT1bmfqYQcDkiJgbEQ8A%0AuwPn92xJjeeZ7CT1GRFxIHAd8J7MvL+n65F6MwNekqQKchO9JEkVZMBLklRBBrwkSRVkwEuSVEEG%0AvFRBEbGqvGJW89+EdZj2sIi4pZPtT4+I0R2cttPtS/JiM1JVvZSZo3qi4fJkNJJ6mD14aT0SEY9H%0AxNfKXv2MiNg7In5enoTozJpRN42IWyPi4Yi4PCI2KKf/QTndnIi4oMV8vx4R9wMn1AzfICImRcRX%0AyvtHRcQ9EXF/RNwYEYPK4cdExEPl9O/tlhdDqjgDXqqmgS020Y+reezJsnf/G2AScDywP3BBzThv%0AAz5JccavXfhH6H4hM0cDI4BDI2JEzTTLMnPvzJxS3u8HXAPMy8xzI2Jr4FzgyMzcG5gBfCoiBgBX%0AAu+iOO/8P3XRayCt19xEL1VTe5vop5X//wQMyszlwPKIeCUiNi8fuy8zHwWIiOuAtwNTgRPL6wr0%0Ao7jG9u7AA+U017do54fADTWnld2/HP93EQGwIXAP8FbgscycV7Z3NcVVvyR1ggEvrX9eKf+/VnO7%0A+X7zZ0LLU1xmRAwD/g+wb2Y+GxGTgAE146xoMc3vgX+OiEsy82UggF9k5sm1I0VEjxwrIFWdm+gl%0AteZtETGs3Pc+DvgtsClFiD8fEdsC71jLPK4CbgNuiIh+wP8AB0XEmwAiYuOIeDPwEDA0InYppzu5%0A1blJWif24KVqGhgRs2ru/3dm1v1TOeAPwKXAm4BfAz/NzNci4o8UgTwf+N3aZpKZ34qIzYD/Ak4B%0APgRcFxEblaOcm5n/W272vzUi/kZxbMAm61CrpFZ4sRlJkirITfSSJFWQAS9JUgUZ8JIkVZABL0lS%0ABRnwkiRVkAEvSVIFGfCSJFWQAS9JUgX9/2tsvjlX0EkjAAAAAElFTkSuQmCC%0A)

    Passengers with missing 'Embarked' values: 2 (2 survived, 0 did not survive)

After exploring the survival statistics visualization, fill in the
missing code below so that the function will make your prediction.\
 Make sure to keep track of the various features and conditions you
tried before arriving at your final prediction model.\
 **Hint:** You can start your implementation of this function using the
prediction code you wrote earlier from `predictions_2`.

In [152]:

    def predictions_3(data):
        """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """    
        predictions = []
        for _, passenger in data.iterrows():
            
            # Remove the 'pass' statement below 
            # and write your prediction conditions here
            if (passenger['Sex'] == 'female'):
                if (passenger['Pclass'] < 3):
                        predictions.append(1)
                elif (passenger['Age'] < 18 and passenger['Pclass']== 3 and passenger['Embarked'] == 'Q'):
                        predictions.append(1)
                elif (passenger['Age'] < 30 and passenger['Pclass']== 3 and passenger['Embarked'] == 'C'):
                        predictions.append(1)
                elif (passenger['Age'] > 50 and passenger['Pclass']== 3 and passenger['Embarked'] == 'S'):
                        predictions.append(1)
                else:
                    predictions.append(0)
            elif (passenger['Sex'] == 'male' and passenger['Age'] < 10):
                predictions.append(1)
            elif (passenger['Sex'] == 'male' and passenger['Age'] < 18):
                if (passenger['Pclass'] < 2) or (passenger['Pclass'] == 2 and passenger['Embarked'] == 'C' ):
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                predictions.append(0)
        
        # Return our predictions
        return pd.Series(predictions)

    # Make the predictions
    predictions = predictions_3(data)

### Question 4[¶](#Question-4) {#Question-4}

*Describe the steps you took to implement the final prediction model so
that it got an accuracy of at least 80%. What features did you look at?
Were certain features more informative than others? Which conditions did
you use to split the survival outcomes in the data? How accurate are
your predictions?*\
 **Hint:** Run the code cell below to see the accuracy of your
predictions.

In [153]:

    print accuracy_score(outcomes, predictions)

    Predictions have an accuracy of 80.81%.

**Answer**: Steps taken to improve prediction model accuracy were:

-   Review data provide in the csv file and filtered for all survived
    (1) passengers
-   Looked at columns that provide more relevance to the survival
    outcome by setting their value and toggling on the survival column
    and selected following features:

> 1.  Sex
> 2.  Age
> 3.  Pclass
> 4.  Embarked

-   Created a pivot table in excel to understand the pattern by
    splitting age in the buckets of 10 years
-   Used vs.survival\_stats to get a visual insight to the data with the
    assumptions tried

Sex, Pclass, Embarked features were more useful due to their discrete
value mapping to the survival outcome.

**Predictions have an accuracy of 80.81%**

Conclusion[¶](#Conclusion) {#Conclusion}
==========================

After several iterations of exploring and conditioning on the data, you
have built a useful algorithm for predicting the survival of each
passenger aboard the RMS Titanic. The technique applied in this project
is a manual implementation of a simple machine learning model, the
*decision tree*. A decision tree splits a set of data into smaller and
smaller groups (called *nodes*), by one feature at a time. Each time a
subset of the data is split, our predictions become more accurate if
each of the resulting subgroups are more homogeneous (contain similar
labels) than before. The advantage of having a computer do things for us
is that it will be more exhaustive and more precise than our manual
exploration above. [This
link](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
provides another introduction into machine learning using a decision
tree.

A decision tree is just one of many models that come from *supervised
learning*. In supervised learning, we attempt to use features of the
data to predict or model things with objective outcome labels. That is
to say, each of our data points has a known outcome value, such as a
categorical, discrete label like `'Survived'`, or a numerical,
continuous value like predicting the price of a house.

### Question 5[¶](#Question-5) {#Question-5}

*Think of a real-world scenario where supervised learning could be
applied. What would be the outcome variable that you are trying to
predict? Name two features about the data used in this scenario that
might be helpful for making the predictions.*

**Answer**: Supervised learning can be applied to determine credit
worthiness of a borrower. The parameters that would play important role
in determining credit worthiness of a borrower could be:

1.  Current credit score
2.  Debt to income ratio with the potential loan

> **Note**: Once you have completed all of the code implementations and
> successfully answered each question above, you may finalize your work
> by exporting the iPython Notebook as an HTML document. You can do this
> by using the menu above and navigating to\
>  **File -\> Download as -\> HTML (.html)**. Include the finished
> document along with this notebook as your submission.
