# TRADE

The TRADE dataset is stored in `TRADE.csv`. The file contains the following columns:
* `image`: an ID identifying the ad image
* `distractor_1`: non-mathing explanation for the ad (hard negative)
* `distractor_2`: non-mathing explanation for the ad (hard negative)
* `matching_explanation`: mathing explanation for the ad as initially introduced in the [Pitt Ads Dataset](https://people.cs.pitt.edu/~mzhang/visualization/dataset/)
* `image_url`: URL of the image as released by the curators of the [Pitt Ads Dataset](https://people.cs.pitt.edu/~mzhang/visualization/dataset/)

The distractor-row indices necessary to construct the 10 different splits of TRADE-control starting from TRADE are stored in `trade_control.json`. The numbers listed within the keys `dist_1` and `dist_2` correspond to the row indexes of the column `matching_explanation` in `TRADE.csv`. Here is an example of how to derive TRADE-control's `split_1` from TRADE:

```
trade = pd.read_csv('TRADE.csv')
control = json.load(open('trade_control.json', 'r'))
trade_control = pd.DataFrame({'matching_explanation': trade.matching_explanation})
trade_control['distractor_1'] = trade_control.matching_explanation.values[control['split_1']['dist1']]
trade_control['distractor_2'] = trade_control.matching_explanation.values[control['split_1']['dist2']]
```

`human_performance.csv` contains the data to reproduce the experiment where we evalauted human performance on TRADE.