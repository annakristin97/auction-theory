## Imports



```python
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import pearsonr
from scipy.stats import zscore
from enum import Enum
from itertools import cycle
from plotly.subplots import make_subplots

import numpy as np
import random as rd
import math
import copy
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

```

## Constants



```python
# plots
colors = ["#A31314", "#2B6999", "#E37002", "#B2C613", "#51A9B0", "#88837D"]
palette = cycle(colors)

# print
new_line = '\n'
new_line_space = '\n' + '   '

```

## Styling Functions



```python
def style_plot(fig):
    layout = {
        'plot_bgcolor': '#ffffff',
        'paper_bgcolor': '#ffffff'
    }
    # Change grid color and axis colors
    fig.update_xaxes(gridcolor='LightGray')
    fig.update_yaxes(gridcolor='LightGray')

    # set white background
    fig.update_layout(layout)

```

# Auction Simulation


## Helper Functions



```python
# returns n values, normally distributed:
#   mean: average value
#   std: standard deviation
def get_normal(mean, std, n):
    return np.random.normal(loc=mean, scale=std, size=n)

```


```python
# returns a random value from a log-normal distribution with
#   mean: average value
#   std: standard deviation
def get_lognormal(mean, std):
    mu = mean
    sigma = std

    a = 1 + (sigma / mu) ** 2
    s = np.sqrt(np.log(a))
    scale = mu / np.sqrt(a)

    return math.floor(lognorm.rvs(s=s, scale=scale))

```


```python
# returns the n-th percentile of a normal distribution with:
#   mean: average value
#   n: n-th percentile

# Ex: 95th percentile -> point which 95% of the numbers are below
def get_nth_percentile(std: float, mean: float, n: int):
    return norm.ppf(n / 100.0, loc=mean, scale=std)  # percent-point-function

```


```python
# returns average of numeric values in a list
def average_value(values: list[int] or list[float]):
    return sum(values) / len(values)

```


```python
# returns:
#   std_self - how little a bidder trusts his original value estimate
#   std_others - how little a bidder trusts other people's bids as estimates
# ..the values are negatively linearly correlated
def calculate_stds(private_info, consensus_bias, desire_coef, risk_coef, std_private_values):
    std_self_coef = average_value(
        [1-max(0, private_info), 1-max(0, consensus_bias), max(0, desire_coef), max(0, risk_coef)])
    std_others_coef = 1 - std_self_coef

    std_self = std_self_coef * std_private_values
    std_others = std_others_coef * std_private_values

    return std_self, std_others

```


```python
# returns distribution of bidder's belief of other people's values as list of floats
def get_value_belief_dist(private_value, std, no_bidders):
    return get_normal(private_value, std, no_bidders)

```


```python
# how much the bidder trusts incoming information at time t
def get_trust_time_coef(t: int):
    return t + math.log(t+1)

```


```python
hist_data = [get_trust_time_coef(t) for t in range(0, 50)]


fig = ff.create_distplot([hist_data], group_labels=['Trust Coefficient'], show_hist=False, show_rug=False,
                         curve_type="kde", bin_size=50, colors=colors)

fig.update_layout(xaxis_title='Time', yaxis_title='Trust Coefficient',
                  title='Bidders Trust in Other Bidders\' Values Through Time')
fig.update_xaxes(range=[0, 25])

style_plot(fig)


fig.show()

```




```python
# Updates a bidders belief set (his value and beliefs for other bidders' values) using:
#   bidder: the bidder whose belief set should be updatet
#   no_bidders: the total number of bidders in the auction
#   time: current time of the auction (discrete counter where one bid = one time unit)
#   all_bids: all bids placed to this point in the auction
def update_belief_set2(bidder, no_bidders, time, all_bids):
    std_incoming = bidder.std_others
    std_prior = bidder.std
    n = time

    # calculate new bidder value
    std_post = math.sqrt(1 /
                         ((1 / math.pow(std_prior, 2)) + (n / math.pow(std_incoming, 2))))

    # calculate new (posterior) belief set parameters
    mu_prior = bidder.curr_value
    x_mean = average_value(list(map(lambda bid: bid.amount, all_bids)))

    mean_post = ((1 / math.pow(std_prior, 2)) / ((1 / math.pow(std_incoming, 2)) + (1 / math.pow(std_prior, 2)))) * mu_prior + \
        ((n / math.pow(std_incoming, 2)) / ((1 / math.pow(std_incoming, 2)) +
         (1 / math.pow(std_prior, 2)))) * x_mean

    # update bidder attributes
    bidder.curr_value = math.floor(mean_post)
    bidder.std = std_post

    # get new belief set for other bidders' values
    bidder.value_belief_distribution = get_value_belief_dist(
        bidder.curr_value, bidder.std, no_bidders)

```

### Plotting



```python
# plots all bidder belief set distributions in one dist plot
def plot_belief_distributions(belief_sets, title):

    hist_data = list(belief_sets)
    group_labels = list(map(lambda x: str(x), range(0, len(belief_sets))))

    fig = ff.create_distplot(hist_data, group_labels, show_hist=False,
                             show_rug=False, curve_type="kde", bin_size=50)
    fig.update_layout(title=title)
    fig.show()

```


```python
# plots all bidder belief set distributions in one scatter graph
def plot_belief_distributions_scatter(belief_sets, title):

    hist_data = list(belief_sets)
    group_labels = list(map(lambda x: str(x), range(0, len(belief_sets))))

    fig = px.scatter(hist_data, color=group_labels, opacity=0.4)
    fig.update_traces(marker={'size': 10})
    fig.update_layout(title=title, width=600, height=800)
    fig.show()

```

## Classes



```python
class Auction:
    def __init__(self, id, N, reserve, min_increment_coef = 0.01):
        self.id = id

        # static values
        self.N = N  # no. bidders
        self.reserve = reserve  # item reserve / auction estimate
        self.bidders = None  # bidders signed up for the auction
        # minimum amount to increment from last bid
        self.min_increment = math.floor(reserve * min_increment_coef)

        # dynamic values
        self.t = 0  # current time
        self.curr_bid = None  # current highest bid
        self.all_bids = []  # all placed bids

    def __str__(self) -> str:

        attribute_strings = (
            'id: ' + self.id + new_line_space +
            'no. bidders: ' + str(self.N) + new_line_space +
            'min_increment: ' + str(self.min_increment) + new_line_space +
            'reserve: ' + str(self.reserve) + new_line
        )

        return (
            'Auction(' + new_line_space +
            attribute_strings +
            ')' + new_line
        )

```


```python

class Bidder:
    def __init__(self, name, predef_value, std_self, std_others, value_belief_distribution):
        self.name = name
        self.predef_value = predef_value  # bidder's estimated value of item pre-auction

        self.curr_value = predef_value  # bidders updated in-auction value
        self.is_active = True  # all bidders start active
        self.no_bids_submitted = 0  # no bids submitted by bidder
        # the maximum amount he will ever update his value to (95th percentile)
        self.max_raise = get_nth_percentile(std_self, predef_value, 95)

        # what he thinks other bidder's values are
        self.value_belief_distribution = value_belief_distribution

        # --- std coefficients ---

        # static
        self.std_self = std_self  # how much bidder trusts his original value estimate
        self.std_others = std_others  # how much the bidder trusts incoming information

        # dynamic
        self.std = std_self  # how much bidder trusts his current value estimate

    def __str__(self) -> str:

        attribute_strings = (
            'name: ' + self.name + new_line_space +
            'predef_value: ' + str(self.predef_value) + new_line_space +
            'curr_value: ' + str(self.curr_value) + new_line_space +
            'max_raise: ' + str(self.max_raise) + new_line_space +
            'std_self: ' + str(self.std_self) + new_line_space +
            'std_others: ' + str(self.std_others) + new_line_space +
            'std: ' + str(self.std) + new_line_space +
            'is_active: ' + str(self.is_active) + new_line
        )

        return (
            'Bidder(' + new_line_space +
            attribute_strings +
            ')' + new_line
        )

```


```python
class Bid:
    def __init__(self, amount: int, bidder: Bidder):
        self.amount = amount  # amount of bid
        self.bidder = bidder  # bidder that placed the bid

    def __str__(self) -> str:
        return 'Bid(amount=' + str(self.amount) + ' ,bidder=' + str(self.bidder) + ')'

```

## Simulation Functions



```python
# returns a bidders bid given
#   curr_bid: the current highest bid
#   curr_time: time passed in the auction
#   bidder: bidder in question
#   no_bidders: total no. bidders in the auction
#   min_increment: minimum increment from current highest bid
def get_bidder_bid(curr_bid: Bid, curr_time: int, bidder: Bidder, no_bidders: int, min_increment: int):
    bid_amount = 0

    # bidder has reached his maximum coming in to the auction
    # OR
    # current bid is higher than his current estimated value
    if ((curr_bid.amount > bidder.max_raise) | (curr_bid.amount > bidder.curr_value)):
        # bidder opts out of the auction and becomes inactive
        bidder.is_active = False
        return 0

    # bidder does not own the current highest bid AND the value + min_increment is still lower than his current value
    if ((curr_bid.bidder != bidder) & ((curr_bid.amount + min_increment) < bidder.curr_value)):
        # bid random on range [current bid + min_increment, value]
        bid_amount = rd.randint(
            curr_bid.amount + min_increment, bidder.curr_value)

    return bid_amount

```


```python

# runs the simulation of a single auction until only one bidder remains
def run_auction(auction):
    # start at time=0 with no bids placed
    auction.curr_bid = Bid(0, None)
    auction.t = 0
    no_more_bids = False

    while (not no_more_bids):
        bids = []

        # collect bids from bidders that are still active
        for bidder in auction.bidders:
            if (bidder.is_active):
                # get proposed bid from bidder
                bid_amount = get_bidder_bid(
                    auction.curr_bid, auction.t, bidder, auction.N, auction.min_increment)

                if (bid_amount > auction.curr_bid.amount):
                    bids.append(Bid(bid_amount, bidder))

        if (len(bids) > 0):
            # grab random bid out of the placed bids at time t and set as current bid
            selected_bid = rd.choice(bids)
            auction.curr_bid = selected_bid
            auction.all_bids.append(selected_bid)

            # update bidder no. bids
            auction.curr_bid.bidder.no_bids_submitted += 1

            # update each active bidder's belief set
            for bidder in auction.bidders:
                if ((bidder.is_active) & (auction.curr_bid.bidder != bidder)):
                    update_belief_set2(
                        bidder=bidder, no_bidders=auction.N, time=auction.t, all_bids=auction.all_bids)

        else:
            # no one wants to bid higher than current bid - end auction
            no_more_bids = True

        auction.t += 1

    return auction.curr_bid

```


```python
# runs n simulations of the auctioning of a lot with
#   estimate: the auction house estimate pre-auction
#   no_bidders: total no. bidders participating
#   affiliation_coef: how affiliated bidder values are pre-auction
def run_simulation(no_iterations, estimate, no_bidders, affiliation_coef=0.05, min_increment_coef=0.01):

    # standard deviation of bidder values - scaled to fit lot estimate
    std = estimate * affiliation_coef

    winning_bids = []
    all_original_bidders = []
    all_final_bidders = []

    for i in range(0, no_iterations):
        # create auction object
        auction = Auction(id='b'+str(i+1), N=no_bidders,
                          reserve=estimate, min_increment_coef=min_increment_coef)

        # create bidder objects
        bidders = []
        for i in range(0, auction.N):
            bidder_private_value = get_lognormal(
                mean=estimate, std=std)
            bidder_private_info = get_normal(
                mean=0.5, std=0.2, n=1)
            bidder_consensus_bias = get_normal(
                mean=0.5, std=0.2, n=1)
            bidder_desire = get_normal(
                mean=0.5, std=0.2, n=1)
            bidder_risk_coef = get_normal(
                mean=0.5, std=0.2, n=1)
            std_self, std_others = calculate_stds(
                bidder_private_info, bidder_consensus_bias, bidder_desire, bidder_risk_coef, std)

            bidders.append(Bidder(
                name='b'+str(i+1),
                predef_value=bidder_private_value,
                std_self=std_self[0],
                std_others=std_others[0],
                value_belief_distribution=get_value_belief_dist(bidder_private_value, std_self, auction.N)))

        # store original bidder attributes
        original_bidders = copy.deepcopy(bidders)
        all_original_bidders.append(original_bidders)

        auction.bidders = bidders

        winning_bids.append(run_auction(auction))
        all_final_bidders.append(auction.bidders)

    return winning_bids, all_final_bidders, all_original_bidders

```

## Simulation



```python
estimate = 1000
no_bidders = 10
winning_bids, all_final_bidders, all_original_bidders = run_simulation(
    500, estimate, no_bidders)

```


```python
all_original_values = []

for auction_bidders in all_original_bidders:
    all_original_values.append(
        list(map(lambda bidder: bidder.predef_value, auction_bidders)))

all_original_values = [
    item for sublist in all_original_values for item in sublist]

fig = ff.create_distplot([all_original_values], group_labels=[
                         'Bidder Original Values'], show_rug=False)
fig.add_vline(x=np.median(all_original_values), line_width=1,
              line_color=next(palette), annotation_text="Median (\"True\" value)", annotation_position="top")

fig.update_layout(title='Distribution of Bidder Values at Auction Start', showlegend=False, xaxis_title='Bidder Values', yaxis_title='Density')

style_plot(fig)
fig.show()

```




```python
# for i in range(0,len(all_original_bidders)):
#     plot_belief_distributions_scatter(list(map(lambda bidder: bidder.value_belief_distribution,
#                                     all_original_bidders[i])), 'Bidder\'s (Original) Belief Distributions')
#     plot_belief_distributions_scatter(list(map(lambda bidder: bidder.value_belief_distribution,
#                                     all_final_bidders[i])), 'Bidder\'s (Final) Belief Sets')

```


```python

# for i in range(0,len(all_original_bidders)):
#     plot_belief_distributions(list(map(lambda bidder: bidder.value_belief_distribution,
#                                  all_original_bidders[i])), 'Bidder\'s (Original) Belief Distributions')
#     plot_belief_distributions(list(map(lambda bidder: bidder.value_belief_distribution,
#                                        all_final_bidders[i])), 'Bidder\'s (Final) Belief Distributions')

```


```python
auction_results = []

for i in range(0, len(winning_bids)):
    winning_bidder = winning_bids[i].bidder
    all_but_winner = filter(lambda bidder: bidder.name !=
                            winning_bidder.name, all_original_bidders[i])
    average_loser_value = average_value(
        list(map(lambda losing_bidder: losing_bidder.curr_value, all_but_winner)))
    winner_curse = average_loser_value - winning_bids[i].amount

    auction_result = {
        'winner_curse': winner_curse,
        'winner_amount': winning_bids[i].amount,
        'winner_utility': winning_bids[i].bidder.curr_value - winning_bids[i].amount,
        'winner_no_bids_submitted': winning_bids[i].bidder.no_bids_submitted,
        'winner_std': winning_bids[i].bidder.std,
        'winner_std_self': winning_bids[i].bidder.std_self,
        'paid_above_reserve': winning_bids[i].amount > estimate
    }
    auction_results.append(auction_result)

df = pd.DataFrame(auction_results)

```


```python
fig = px.scatter(df, x="winner_amount", y="winner_curse",
                 opacity=0.3, title='Simulation: Winner\'s Curse vs. Amount Paid', color='paid_above_reserve', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_xaxes(title='Amount Paid for Item')
fig.update_yaxes(title='Average Loser Value - Amount Paid')
style_plot(fig)
fig.show()

```




```python
fig = px.scatter(df, x="winner_amount", y="winner_utility", trendline='ols',
                 opacity=0.3, title='Simulation: Winner Utility vs. Amount Paid', color_discrete_sequence=sorted(colors))
fig.update_traces(marker_size=5)
fig.update_xaxes(title='Amount Paid for Item')
fig.update_yaxes(title='Winner Value - Amount Paid')
style_plot(fig)
fig.show()

```




```python
# ---WINNING BIDDER STD_SELF vs. STD_FINAL---
fig = px.scatter(df, x="winner_std", y="winner_std_self",
                 opacity=0.3, title='Simulation: Winner Initial vs. Final Trust', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_xaxes(title='Final Trust in Estimate')
fig.update_yaxes(title='Initial Trust in Own Estimate')
style_plot(fig)
fig.show()

```




```python
# ---WINNING BIDDER NO BIDS SUBMITTED---
fig = px.histogram(map(lambda no: str(
    no), df['winner_no_bids_submitted'].sort_values()), color_discrete_sequence=colors[2:])

fig.update_xaxes(title='No. Bids Submitted')
fig.update_yaxes(title='Count')
fig.update_layout(title='Simulation: No. Bids Submitted by Winner During Auction', showlegend=False)
style_plot(fig)
fig.show()

```




```python
# ---WINNING BIDDER NO_BIDS vs. PRICE---
fig = px.scatter(df, y="winner_no_bids_submitted", x="winner_amount",
                 opacity=0.3, title='Simulation: Price Paid vs. Winner No. Bids Submitted', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_yaxes(title='No. Bids Submitted')
fig.update_xaxes(title='Price Paid')
style_plot(fig)
fig.show()

```



## Real Data (Sotheby's)



```python
df = pd.read_csv('auction_data.csv')
df_non_null = df[df['number_of_bidders'].notnull()]
```


```python

fig = px.scatter(x=df_non_null['low_estimate'], y=df_non_null['current_bid'],
                 opacity=0.6, trendline='ols', log_x=True, log_y=True, color_discrete_sequence=colors)
fig.update_traces(marker={'size': 3})
fig.update_yaxes(title='Final Price (log)')
fig.update_xaxes(title='Low Estimate (log)')
fig.update_layout(title='Sotheby\'s: Low Estimate vs. Final Price', width=1000, height=600)
style_plot(fig)
fig.show()

```




```python
fig = px.scatter(x=df_non_null['number_of_bids'], y=df_non_null['low_estimate'] -
                 df_non_null['current_bid'], opacity=0.5, trendline='ols', color_discrete_sequence=sorted(colors))
fig.update_traces(marker={'size': 4})
fig.update_yaxes(title='Low Estimate - Final Price')
fig.update_xaxes(title='No. Bids')
fig.update_layout(
    title='Sotheby\'s: No. Bids vs. Low Estimate - Final Price', width=1000, height=600)
    
style_plot(fig)
fig.show()

```



### Version 2



```python
df = pd.read_csv('auction_data_v2.csv')
df_non_null = df[df['number_of_bidders'].notnull()]
```

### No. Bids Placed by Winner


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_time</th>
      <th>event_name</th>
      <th>auction_id</th>
      <th>lot_id</th>
      <th>low_estimate</th>
      <th>reserve</th>
      <th>number_of_bids</th>
      <th>current_bid</th>
      <th>paddle</th>
      <th>number_of_bidders</th>
      <th>sale_end_date</th>
      <th>auction_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009-10-17T19:21:01.059000</td>
      <td>BiddenOnLotEvent</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2000.0</td>
      <td>1200.0</td>
      <td>1</td>
      <td>1000.0</td>
      <td>177.0</td>
      <td>1</td>
      <td>2009-10-30T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009-10-17T19:21:01.154000</td>
      <td>ConsignorBidEvent</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2000.0</td>
      <td>1200.0</td>
      <td>2</td>
      <td>1100.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>2009-10-30T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009-10-17T19:21:25.624000</td>
      <td>BiddenOnLotEvent</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2000.0</td>
      <td>1200.0</td>
      <td>3</td>
      <td>1200.0</td>
      <td>177.0</td>
      <td>1</td>
      <td>2009-10-30T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009-10-19T21:11:52.524000</td>
      <td>BiddenOnLotEvent</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2000.0</td>
      <td>1200.0</td>
      <td>4</td>
      <td>1300.0</td>
      <td>366.0</td>
      <td>2</td>
      <td>2009-10-30T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009-10-19T21:11:52.587000</td>
      <td>ReactiveBidEventV2</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2000.0</td>
      <td>1200.0</td>
      <td>5</td>
      <td>1400.0</td>
      <td>366.0</td>
      <td>2</td>
      <td>2009-10-30T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>33786</th>
      <td>2009-10-07T14:36:54.463000</td>
      <td>ReactiveBidEventV2</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>14</td>
      <td>1600.0</td>
      <td>621.0</td>
      <td>8</td>
      <td>2009-10-09T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>33787</th>
      <td>2009-10-07T14:37:31.275000</td>
      <td>BiddenOnLotEvent</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>15</td>
      <td>1700.0</td>
      <td>621.0</td>
      <td>8</td>
      <td>2009-10-09T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>33788</th>
      <td>2009-10-07T14:37:31.361000</td>
      <td>ReactiveBidEventV2</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>16</td>
      <td>1800.0</td>
      <td>621.0</td>
      <td>8</td>
      <td>2009-10-09T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>33789</th>
      <td>2009-10-07T14:37:56.789000</td>
      <td>BiddenOnLotEvent</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>17</td>
      <td>2000.0</td>
      <td>621.0</td>
      <td>8</td>
      <td>2009-10-09T04:00:00</td>
      <td>TIMED</td>
    </tr>
    <tr>
      <th>33790</th>
      <td>2009-10-07T14:37:56.878000</td>
      <td>ReactiveBidEventV2</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>500.0</td>
      <td>500.0</td>
      <td>18</td>
      <td>2000.0</td>
      <td>621.0</td>
      <td>8</td>
      <td>2009-10-09T04:00:00</td>
      <td>TIMED</td>
    </tr>
  </tbody>
</table>
<p>33791 rows × 12 columns</p>
</div>




```python
agg_map = {'current_bid': ['count', 'max'], 'low_estimate': 'max'}

df_paddle_ids = df.groupby(['auction_id', 'lot_id', 'paddle']).agg(
    agg_map).reset_index()

# unnest dataframe columns
df_paddle_ids.columns = df_paddle_ids.columns.map('_'.join)
df_paddle_ids = df_paddle_ids.reset_index()

df_paddle_ids = df_paddle_ids.rename(columns={'current_bid_count': 'bid_count', 'current_bid_max': 'bid_max'})

agg_map2 = {'bid_max': 'max'}

# df_paddle_ids = df.groupby(['auction_id', 'lot_id']).agg(
#     agg_map).reset_index()

df_paddle_ids = df_paddle_ids[df_paddle_ids.groupby(['auction_id_', 'lot_id_'])['bid_max'].transform(max) == df_paddle_ids['bid_max']]

df_paddle_ids = df_paddle_ids.rename(columns={'auction_id_': 'auction_id', 'lot_id_': 'lot_id', 'paddle_': 'paddle', 'low_estimate_max': 'low_estimate'})

# # unnest dataframe columns
# df_paddle_ids.columns = df_paddle_ids.columns.map('_'.join)
# df_paddle_ids = df_paddle_ids.reset_index()


# df_paddle_ids = df_paddle_ids.rename(columns={'current_bid_count': 'bid_count', 'current_bid_max': 'bid_max', 'low_estimate_max': 'low_estimate'})

#add custom columns
df_paddle_ids['est_winbid_diff'] = df_paddle_ids['low_estimate'] - \
    df_paddle_ids['bid_max']
df_paddle_ids['perc_above_estimate'] = (df_paddle_ids['bid_max'] -
                                         df_paddle_ids['low_estimate']) / df_paddle_ids['low_estimate']

# remove outliers
df_paddle_ids_no_outliers = df_paddle_ids[(
np.abs(zscore(df_paddle_ids.select_dtypes(include=np.number))) < 3).all(axis=1)]

df_paddle_ids

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>auction_id</th>
      <th>lot_id</th>
      <th>paddle</th>
      <th>bid_count</th>
      <th>bid_max</th>
      <th>low_estimate</th>
      <th>est_winbid_diff</th>
      <th>perc_above_estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>543.0</td>
      <td>3</td>
      <td>2400.0</td>
      <td>2000.0</td>
      <td>-400.0</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>-8948241167762729875</td>
      <td>-9194564856127871092</td>
      <td>918.0</td>
      <td>5</td>
      <td>1400.0</td>
      <td>500.0</td>
      <td>-900.0</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>-8948241167762729875</td>
      <td>-9151072587192145600</td>
      <td>822.0</td>
      <td>1</td>
      <td>900.0</td>
      <td>1200.0</td>
      <td>300.0</td>
      <td>-0.250000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>-8948241167762729875</td>
      <td>-9123971500861958519</td>
      <td>714.0</td>
      <td>2</td>
      <td>300.0</td>
      <td>500.0</td>
      <td>200.0</td>
      <td>-0.400000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>-8948241167762729875</td>
      <td>-9102446430804671452</td>
      <td>609.0</td>
      <td>4</td>
      <td>700.0</td>
      <td>400.0</td>
      <td>-300.0</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>11416</th>
      <td>11416</td>
      <td>3131899225618458582</td>
      <td>2913979777108127063</td>
      <td>1092.0</td>
      <td>2</td>
      <td>4000.0</td>
      <td>4000.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11417</th>
      <td>11417</td>
      <td>3131899225618458582</td>
      <td>3319739171358374966</td>
      <td>525.0</td>
      <td>1</td>
      <td>1400.0</td>
      <td>1500.0</td>
      <td>100.0</td>
      <td>-0.066667</td>
    </tr>
    <tr>
      <th>11418</th>
      <td>11418</td>
      <td>3131899225618458582</td>
      <td>3341618596499002970</td>
      <td>702.0</td>
      <td>8</td>
      <td>4800.0</td>
      <td>3000.0</td>
      <td>-1800.0</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>11421</th>
      <td>11421</td>
      <td>3131899225618458582</td>
      <td>3377077189527049786</td>
      <td>1026.0</td>
      <td>1</td>
      <td>50000.0</td>
      <td>60000.0</td>
      <td>10000.0</td>
      <td>-0.166667</td>
    </tr>
    <tr>
      <th>11429</th>
      <td>11429</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>621.0</td>
      <td>6</td>
      <td>2000.0</td>
      <td>500.0</td>
      <td>-1500.0</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>2903 rows × 9 columns</p>
</div>




```python
# ---WINNING BIDDER NO_BIDS vs. PRICE---
fig = px.scatter(df_paddle_ids_no_outliers, x="perc_above_estimate", y="bid_count",
                 opacity=0.3, title='Sotheby\'s: Price Paid Above Estimate vs. Winner No. Bids Submitted', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_yaxes(title='No. Bids Submitted')
fig.update_xaxes(title='Percentage Paid Above Estimate')
style_plot(fig)
fig.show()
```




```python
# returns second highest value in column
def second_max_func(x):
    y = np.sort(x)
    return y[-2] if len(y) > 1 else x

```


```python
agg_map = {'current_bid': ['max', second_max_func], 'low_estimate': 'max',
           'number_of_bids': 'max', 'number_of_bidders': 'max'}

df_lot_results = df.groupby(['auction_id', 'lot_id']).agg(
    agg_map).reset_index().rename(columns={'current_bid': 'bids'})

# unnest dataframe columns
df_lot_results.columns = df_lot_results.columns.map('_'.join)
df_lot_results = df_lot_results.reset_index()

# rename columns
df_lot_results = df_lot_results.rename(columns={"bids_max": "winning_bid", "bids_second_max_func": "bids_second_max",
                                       "low_estimate_max": "low_estimate", "number_of_bids_max": "number_of_bids", "number_of_bidders_max": "number_of_bidders"})


df_lot_results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>auction_id_</th>
      <th>lot_id_</th>
      <th>winning_bid</th>
      <th>bids_second_max</th>
      <th>low_estimate</th>
      <th>number_of_bids</th>
      <th>number_of_bidders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2400.0</td>
      <td>2200.0</td>
      <td>2000.0</td>
      <td>15</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-8948241167762729875</td>
      <td>-9194564856127871092</td>
      <td>1400.0</td>
      <td>1300.0</td>
      <td>500.0</td>
      <td>11</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-8948241167762729875</td>
      <td>-9151072587192145600</td>
      <td>900.0</td>
      <td>800.0</td>
      <td>1200.0</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-8948241167762729875</td>
      <td>-9123971500861958519</td>
      <td>300.0</td>
      <td>200.0</td>
      <td>500.0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-8948241167762729875</td>
      <td>-9102446430804671452</td>
      <td>700.0</td>
      <td>600.0</td>
      <td>400.0</td>
      <td>8</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2831</th>
      <td>2831</td>
      <td>3131899225618458582</td>
      <td>2913979777108127063</td>
      <td>4000.0</td>
      <td>3800.0</td>
      <td>4000.0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2832</th>
      <td>2832</td>
      <td>3131899225618458582</td>
      <td>3319739171358374966</td>
      <td>1400.0</td>
      <td>1400.0</td>
      <td>1500.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2833</th>
      <td>2833</td>
      <td>3131899225618458582</td>
      <td>3341618596499002970</td>
      <td>4800.0</td>
      <td>4500.0</td>
      <td>3000.0</td>
      <td>11</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2834</th>
      <td>2834</td>
      <td>3131899225618458582</td>
      <td>3377077189527049786</td>
      <td>50000.0</td>
      <td>50000.0</td>
      <td>60000.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>2835</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>500.0</td>
      <td>18</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>2836 rows × 8 columns</p>
</div>




```python
# add custom columns
df_lot_results['est_winbid_diff'] = df_lot_results['low_estimate'] - \
    df_lot_results['winning_bid']
df_lot_results['paid_above_estimate'] = df_lot_results['winning_bid'] > \
    df_lot_results['low_estimate']
df_lot_results['perc_above_estimate'] = (df_lot_results['low_estimate'] -
                                         df_lot_results['winning_bid']) / df_lot_results['low_estimate']
df_lot_results['prop_max_secondmax_diff'] = (df_lot_results['winning_bid'] -
                                             df_lot_results['bids_second_max']) / df_lot_results['winning_bid']

```


```python
# remove outliers
df_lot_results_no_outliers = df_lot_results[(
    np.abs(zscore(df_lot_results.select_dtypes(include=np.number))) < 3).all(axis=1)]

```


```python
# --- Winning Bid vs. Low Estimate ---
fig = px.scatter(df_lot_results_no_outliers, x="low_estimate", y="winning_bid", color="paid_above_estimate",
                 opacity=0.3, title='Sotheby\'s: Winner\'s Curse?', log_x=True, log_y=True, color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_layout(yaxis_title='Winning Bid', xaxis_title='(Low) Estimate')
fig.update_layout(shapes=[{'type': 'line', 'y0': df_lot_results_no_outliers['winning_bid'].min(), 'y1': df_lot_results_no_outliers['winning_bid'].max(
), 'x0': df_lot_results_no_outliers['winning_bid'].min(), 'x1': df_lot_results_no_outliers['winning_bid'].max()}])

style_plot(fig)
fig.show()

```




```python
# --- Winning Bid vs. Second-highest Bid ---
fig = px.scatter(df_lot_results_no_outliers, x="winning_bid", y="prop_max_secondmax_diff", color="paid_above_estimate",
                 opacity=0.3, title='Sotheby\'s: Winner\'s Curse?', log_x=True, color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Winning Bid',
                  yaxis_title='Last Raise % by Winner')

style_plot(fig)
fig.show()

```




```python
lots_gone_under_est = df_lot_results.loc[df_lot_results['winning_bid']
                                         < df_lot_results['low_estimate']]

perc_lots_under_est = len(lots_gone_under_est) / len(df_lot_results)
print('Ratio of lots gone under low estimate: ',
      format(perc_lots_under_est, '.2f'))

```

    Ratio of lots gone under low estimate:  0.24



```python
# --- No. Bidders vs. Amount Over Estimate ---
fig = px.scatter(df_lot_results_no_outliers, x="number_of_bidders", y="est_winbid_diff",
                 opacity=0.3, title='Sotheby\'s: No. Unique Bidders vs. Estimate - Winning Bid', trendline='ols')
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Number of Unique Bidders',
                  yaxis_title='Low Estimate - Winning Bid',
                  height=800,
                  width=1000)
style_plot(fig)
fig.show()

```



## Compare Simulation Results to Actual Results



```python
df_lot_results_no_outliers

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>auction_id_</th>
      <th>lot_id_</th>
      <th>winning_bid</th>
      <th>bids_second_max</th>
      <th>low_estimate</th>
      <th>number_of_bids</th>
      <th>number_of_bidders</th>
      <th>est_winbid_diff</th>
      <th>paid_above_estimate</th>
      <th>perc_above_estimate</th>
      <th>prop_max_secondmax_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-8948241167762729875</td>
      <td>-9196689798081083921</td>
      <td>2400.0</td>
      <td>2200.0</td>
      <td>2000.0</td>
      <td>15</td>
      <td>7</td>
      <td>-400.0</td>
      <td>True</td>
      <td>-0.200000</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-8948241167762729875</td>
      <td>-9194564856127871092</td>
      <td>1400.0</td>
      <td>1300.0</td>
      <td>500.0</td>
      <td>11</td>
      <td>3</td>
      <td>-900.0</td>
      <td>True</td>
      <td>-1.800000</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-8948241167762729875</td>
      <td>-9151072587192145600</td>
      <td>900.0</td>
      <td>800.0</td>
      <td>1200.0</td>
      <td>4</td>
      <td>3</td>
      <td>300.0</td>
      <td>False</td>
      <td>0.250000</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-8948241167762729875</td>
      <td>-9102446430804671452</td>
      <td>700.0</td>
      <td>600.0</td>
      <td>400.0</td>
      <td>8</td>
      <td>3</td>
      <td>-300.0</td>
      <td>True</td>
      <td>-0.750000</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>-8948241167762729875</td>
      <td>-9099861984859344664</td>
      <td>400.0</td>
      <td>300.0</td>
      <td>400.0</td>
      <td>4</td>
      <td>2</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2831</th>
      <td>2831</td>
      <td>3131899225618458582</td>
      <td>2913979777108127063</td>
      <td>4000.0</td>
      <td>3800.0</td>
      <td>4000.0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>False</td>
      <td>0.000000</td>
      <td>0.050000</td>
    </tr>
    <tr>
      <th>2832</th>
      <td>2832</td>
      <td>3131899225618458582</td>
      <td>3319739171358374966</td>
      <td>1400.0</td>
      <td>1400.0</td>
      <td>1500.0</td>
      <td>1</td>
      <td>1</td>
      <td>100.0</td>
      <td>False</td>
      <td>0.066667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2833</th>
      <td>2833</td>
      <td>3131899225618458582</td>
      <td>3341618596499002970</td>
      <td>4800.0</td>
      <td>4500.0</td>
      <td>3000.0</td>
      <td>11</td>
      <td>3</td>
      <td>-1800.0</td>
      <td>True</td>
      <td>-0.600000</td>
      <td>0.062500</td>
    </tr>
    <tr>
      <th>2834</th>
      <td>2834</td>
      <td>3131899225618458582</td>
      <td>3377077189527049786</td>
      <td>50000.0</td>
      <td>50000.0</td>
      <td>60000.0</td>
      <td>1</td>
      <td>1</td>
      <td>10000.0</td>
      <td>False</td>
      <td>0.166667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>2835</td>
      <td>3131899225618458582</td>
      <td>3491731720948698270</td>
      <td>2000.0</td>
      <td>2000.0</td>
      <td>500.0</td>
      <td>18</td>
      <td>8</td>
      <td>-1500.0</td>
      <td>True</td>
      <td>-3.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>2617 rows × 12 columns</p>
</div>




```python
actual_results = []
simulation_averages = []
actual_simulation_diffs = []


for idx, lot_row in df_lot_results_no_outliers.iterrows():
    estimate = lot_row['low_estimate']
    no_bidders = lot_row['number_of_bidders']
    winning_bids, all_final_bidders, all_original_bidders = run_simulation(
        100, estimate, no_bidders)

    simulation_avg_winning_bid = average_value(
        list(map(lambda bid: bid.amount, winning_bids)))

    actual_results.append(lot_row['winning_bid'])
    simulation_averages.append(simulation_avg_winning_bid)
    actual_simulation_diffs.append(
        (abs(lot_row['winning_bid'] - simulation_avg_winning_bid) / lot_row['winning_bid']))

```


```python
# --- Winning Bid - Actual vs. Simulation ---
fig = px.scatter(x=actual_results, y=simulation_averages, log_x=True, log_y=True,
                 opacity=0.3, title='Winning Bid - Actual vs. Simulation', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Actual Result',
                  yaxis_title='Simulation Averages')

style_plot(fig)
fig.show()

```




```python
# --- Simulation error vs. winning price ---
fig = px.scatter(x=actual_results, y=actual_simulation_diffs, log_x=True,
                 opacity=0.3, title='Simulation error vs. winning price', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Actual Result',
                  yaxis_title='Simulation Error (%)')

style_plot(fig)
fig.show()

```




```python
# statistic: measure of the strength and direction of association that exists between two variables measured
# pvalue: p-value (very low) suggests that the correlation coefficient is statistically significant,
# being much less than 0.01 (0.01 ---> the risk of concluding that a correlation exists when, actually,
#  no correlation exists is 1%).
print('Actual vs. Estimate: ')
print(pearsonr(df_lot_results_no_outliers['winning_bid'],
      df_lot_results_no_outliers['low_estimate']))

print('Actual vs. Simulation: ')
print(pearsonr(actual_results, simulation_averages))

print('Simulation vs. Estimate: ')
print(pearsonr(simulation_averages,
      df_lot_results_no_outliers['low_estimate']))

```

    Actual vs. Estimate: 
    PearsonRResult(statistic=0.8827620288476193, pvalue=0.0)
    Actual vs. Simulation: 
    PearsonRResult(statistic=0.9126465863200122, pvalue=0.0)
    Simulation vs. Estimate: 
    PearsonRResult(statistic=0.9650728482969742, pvalue=0.0)



```python
# --- Estimate error vs. winning price ---
estimate_diff_prop = abs(
    (df_lot_results_no_outliers['winning_bid'] - df_lot_results_no_outliers['low_estimate'])) / df_lot_results_no_outliers.head(500)['winning_bid']

fig = px.scatter(x=actual_results, y=estimate_diff_prop, log_x=True,
                 opacity=0.3, title='Estimation error vs. winning price', color_discrete_sequence=colors)
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Winning Bid',
                  yaxis_title='Estimation Error (%)')

style_plot(fig)
fig.show()

```



## Parameter Tweaking


### Number of Bidders



```python
bidder_count_avgs = []

for bidder_count in range(2, 20):
    winning_bids, all_final_bidders, all_original_bidders = run_simulation(
        500, 10000, bidder_count)

    simulation_avg_winning_bid = average_value(
        list(map(lambda bid: bid.amount, winning_bids)))

    bidder_count_avgs.append(
        {
            'bidder_count': bidder_count,
            'winning_avg': simulation_avg_winning_bid
        }
    )

```


```python
# --- No. Bidders vs. Winning Price ---
fig = px.scatter(bidder_count_avgs, x="bidder_count", y="winning_avg",
                 opacity=0.5, title='Simulation: Winner\'s Curse and Number of Biddders')
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Number of Bidders',
                  yaxis_title='Winning Bid',
                  height=600,
                  width=1000)

style_plot(fig)
fig.show()

```



### Item Price



```python
price_avgs = []

for price in [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
    winning_bids, all_final_bidders, all_original_bidders = run_simulation(
        500, price, 10)

    simulation_avg_winning_bid = average_value(
        list(map(lambda bid: bid.amount, winning_bids)))

    price_avgs.append(
        {
            'price': price,
            'bid_med_diff': ((simulation_avg_winning_bid - price) / price) * 100
        }
    )

```


```python
# --- Price vs. Winning Bid ---
fig = px.scatter(price_avgs, x="price", y="bid_med_diff",
                 opacity=0.5, title='Simulation: Winner\'s Curse and Item Price', log_x=True)
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Estimate (log)',
                  yaxis_title='Winning Bid Average % Increase',
                  height=600,
                  width=1000)

style_plot(fig)
fig.show()

```



### Bidder Value Affiliation



```python
price_avgs = []
original_values_aff = []

row = 0

fig = make_subplots(rows=5, shared_xaxes=True)
for affiliation in [0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]:

    winning_bids, all_final_bidders, all_original_bidders = run_simulation(
        500, 10000, 10, affiliation_coef=affiliation)

    simulation_avg_winning_bid = average_value(
        list(map(lambda bid: bid.amount, winning_bids)))

    all_original_values = []

    for auction_bidders in all_original_bidders:
        all_original_values.append(
            list(map(lambda bidder: bidder.predef_value, auction_bidders)))

    all_original_values = [
        item for sublist in all_original_values for item in sublist]

    price_avgs.append(
        {
            'affiliation': affiliation,
            'winning_bid': simulation_avg_winning_bid
        }
    )

    if (affiliation in [0.01, 0.05, 0.1, 0.15, 0.2]):
        row += 1
        fig.append_trace(go.Histogram(x=all_original_values,
                         histfunc='count', name=affiliation), row=row, col=1)


style_plot(fig)
fig.update_layout(width=800, height=600,
                  title='Simulation: Bidder Value Distribution With Different Affiliation Coefficients')
fig.update_yaxes(showticklabels=False)
fig.show()

```




```python
# --- Affiliation Coefficient vs. Winning Bid ---
fig = px.scatter(price_avgs, x="affiliation", y="winning_bid",
                 opacity=0.5, title='Simulation: Winner\'s Curse and Bidder Value Affiliation')
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Affiliation',
                  yaxis_title='Winning Bid',
                  height=600,
                  width=1000)

style_plot(fig)
fig.show()

```



### Minimum Increment


```python
price_avgs = []
price = 10000

for min_inc in [x * 0.001 for x in range(1,30)]:
    winning_bids, all_final_bidders, all_original_bidders = run_simulation(
        1000, price, 10, min_increment_coef=min_inc)

    simulation_avg_winning_bid = average_value(
        list(map(lambda bid: bid.amount, winning_bids)))

    price_avgs.append(
        {
            'min_inc': min_inc,
            'bid_med_diff': ((simulation_avg_winning_bid - price) / price) * 100
        }
    )

```


```python
# --- Affiliation Coefficient vs. Winning Bid ---
fig = px.scatter(price_avgs, x="min_inc", y="bid_med_diff",
                 opacity=0.5, title='Simulation: Winner\'s Curse and Minimum Increment')
fig.update_traces(marker_size=5)
fig.update_layout(xaxis_title='Minimum Increment as % of Lot Estimate',
                  yaxis_title='Winning Bid Average % Increase from Estimate',
                  height=600,
                  width=1000)

style_plot(fig)
fig.show()
```




```python

```
