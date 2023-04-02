from datetime import datetime
import numpy as np
import pandas as pd
import pyvinecopulib as pv
import random

date_format = "%Y-%m-%d"

def fitShiftedAnnually(stocks, market, stock_info, family_set, window_size_years):

    controls = pv.FitControlsBicop(family_set = family_set)
    display(controls)

    # empty dictionary
    copula_data_final = {'symbol':[], 'year':[], 'family':[], 'rotation':[], 'tau':[], 'parameters':[]}

    # fiscal year dates mapping
    fiscal_year = {12: ("-01-01","-12-31"),
                  11: ("-12-01", "-11-30"),
                  10: ("-11-01", "-10-31"),
                   9: ("-10-01", "-09-30"),
                   8: ("-09-01", "-08-31"),
                   7: ("-08-01", "-07-31"),
                   6: ("-07-01", "-06-30"),
                   5: ("-06-01", "-05-31"),
                   4: ("-05-01", "-04-30"),
                   3: ("-04-01", "-03-31"),
                   2: ("-03-01", "-02-28"),
                   1: ("-02-01", "-01-31")
                  }

    # symbol list for slicing
    full_symbol_list = stocks['symbol'].drop_duplicates().tolist()

    # slices dataframe per each stock symbol
    for symbol in full_symbol_list:

        # get fiscal year end month to slice accordingly
        stock = stocks[stocks['symbol'] == symbol]
        try:
            month = int(stock_info[stock_info['symbol'] == symbol]['fy_end_date'])
        except:
            month = 0

        if month == 0:
            continue

        year = 2000

        # slices stocksymbol dataframe per each year
        for j in range(22):

            prev_year = year-1

            start, end = fiscal_year.get(month)

            # date string builder
            if month == 12:
                start = str(year - window_size_years + 1) + start
                end = str(year) + end
            else:
                start = str(prev_year - window_size_years + 1) + start
                end = str(year) + end

            # window slice of stock
            stock_year = stock[(stock['date'] >= start) & (stock['date'] <= end)]

            if stock_year.empty:
                pass
            # no copula fitting if less than 200 observations per year
            elif len(stock_year) < 200:
                pass
            else:
                market_year  = market[(market['date'] >= start) & (market['date'] <= end)]
                merged_data = pd.merge(stock_year, market_year, on=['date'], how='inner')
                merged_data = merged_data.drop(columns=['symbol','date'])

                # pseudo observation transformation
                pseudo_merged_data = pv.to_pseudo_obs(merged_data)

                # fit copula to data
                copula = pv.Bicop(data = pseudo_merged_data, controls = controls)

                copula_data_final['symbol'].append(symbol)
                copula_data_final['year'].append(year)
                copula_data_final['family'].append(copula.family)
                copula_data_final['rotation'].append(copula.rotation)
                copula_data_final['tau'].append(copula.tau)
                copula_data_final['parameters'].append(copula.parameters)

            year+=1

    copula_data_final = pd.DataFrame(copula_data_final)
    copula_data_final = copula_data_final.sort_values(by = ['symbol', 'year'])
    copula_data_final = copula_data_final.reset_index(drop =  True)

    return copula_data_final

def fitShiftedAnnuallyTest(stocks, market, stock_info, family_set, window_size_years, ordered_stocks_amount):

    controls = pv.FitControlsBicop(family_set = family_set)
    display(controls)

    # empty dictionary
    copula_data_final = {'symbol':[], 'year':[], 'family':[], 'rotation':[], 'tau':[], 'parameters':[]}

    # fiscal year dates mapping
    fiscal_year = {12: ("-01-01","-12-31"),
                  11: ("-12-01", "-11-30"),
                  10: ("-11-01", "-10-31"),
                   9: ("-10-01", "-09-30"),
                   8: ("-09-01", "-08-31"),
                   7: ("-08-01", "-07-31"),
                   6: ("-07-01", "-06-30"),
                   5: ("-06-01", "-05-31"),
                   4: ("-05-01", "-04-30"),
                   3: ("-04-01", "-03-31"),
                   2: ("-03-01", "-02-28"),
                   1: ("-02-01", "-01-31")
                  }

    # symbol list for slicing
    full_symbol_list = stocks['symbol'].drop_duplicates().tolist()

    # slices dataframe per each stock symbol
    for symbol in full_symbol_list[:ordered_stocks_amount]:
        print(str(symbol) +  ' start')

        # get fiscal year end month to slice accordingly
        stock = stocks[stocks['symbol'] == symbol]
        try:
            month = int(stock_info[stock_info['symbol'] == symbol]['fy_end_date'])
        except:
            month = 0

        if month == 0:
            continue

        year = 2000

        # slices stocksymbol dataframe per each year
        for j in range(22):

            prev_year = year-1

            start, end = fiscal_year.get(month)

            # date string builder
            if month == 12:
                start = str(year - window_size_years + 1) + start
                end = str(year) + end
            else:
                start = str(prev_year - window_size_years + 1) + start
                end = str(year) + end

            # one-year slice of stock
            stock_year = stock[(stock['date'] >= start) & (stock['date'] <= end)]

            if stock_year.empty:
                pass
            # no copula fitting if less than 200 observations per year
            elif len(stock_year) < 200:
                pass
            else:
                market_year  = market[(market['date'] >= start) & (market['date'] <= end)]
                merged_data = pd.merge(stock_year, market_year, on=['date'], how='inner')
                merged_data = merged_data.drop(columns=['symbol','date'])

                # pseudo observation transformation
                pseudo_merged_data = pv.to_pseudo_obs(merged_data)

                # fit copula to data
                copula = pv.Bicop(data = pseudo_merged_data, controls = controls)

                copula_data_final['symbol'].append(symbol)
                copula_data_final['year'].append(year)
                copula_data_final['family'].append(copula.family)
                copula_data_final['rotation'].append(copula.rotation)
                copula_data_final['tau'].append(copula.tau)
                copula_data_final['parameters'].append(copula.parameters)

            year+=1
        print(str(symbol) +  ' done')
        print()

    copula_data_final = pd.DataFrame(copula_data_final)
    copula_data_final = copula_data_final.sort_values(by = ['symbol', 'year'])
    copula_data_final = copula_data_final.reset_index(drop =  True)

    return copula_data_final

def fitShiftedAnnuallyRandTest(stocks, market, stock_info, family_set,window_size_years, random_stocks_amount):

    controls = pv.FitControlsBicop(family_set = family_set)
    display(controls)

    # empty dictionary
    copula_data_final = {'symbol':[], 'year':[], 'family':[], 'rotation':[], 'tau':[], 'parameters':[]}

    # fiscal year dates mapping
    fiscal_year = {12: ("-01-01","-12-31"),
                  11: ("-12-01", "-11-30"),
                  10: ("-11-01", "-10-31"),
                   9: ("-10-01", "-09-30"),
                   8: ("-09-01", "-08-31"),
                   7: ("-08-01", "-07-31"),
                   6: ("-07-01", "-06-30"),
                   5: ("-06-01", "-05-31"),
                   4: ("-05-01", "-04-30"),
                   3: ("-04-01", "-03-31"),
                   2: ("-03-01", "-02-28"),
                   1: ("-02-01", "-01-31")
                  }

    # symbol list for slicing
    full_symbol_list = stocks['symbol'].drop_duplicates().tolist()
    # randomized for testing (set k for sample length)
    sample_symbol_list = random.choices(full_symbol_list, k = random_stocks_amount)

    # slices dataframe per each stock symbol
    for symbol in sample_symbol_list:
        print(str(symbol) +  ' start')

        # get fiscal year end month to slice accordingly
        stock = stocks[stocks['symbol'] == symbol]
        try:
            month = int(stock_info[stock_info['symbol'] == symbol]['fy_end_date'])
        except:
            month = 0

        if month == 0:
            continue

        year = 2000

        # slices stocksymbol dataframe per each year
        for j in range(22):

            prev_year = year-1

            start, end = fiscal_year.get(month)

            # date string builder
            if month == 12:
                start = str(year - window_size_years + 1) + start
                end = str(year) + end
            else:
                start = str(prev_year - window_size_years + 1) + start
                end = str(year) + end

            # one-year slice of stock
            stock_year = stock[(stock['date'] >= start) & (stock['date'] <= end)]

            if stock_year.empty:
                pass
            # no copula fitting if less than 200 observations per year
            elif len(stock_year) < 200:
                pass
            else:
                market_year  = market[(market['date'] >= start) & (market['date'] <= end)]
                merged_data = pd.merge(stock_year, market_year, on=['date'], how='inner')
                merged_data = merged_data.drop(columns=['symbol','date'])

                # pseudo observation transformation
                pseudo_merged_data = pv.to_pseudo_obs(merged_data)

                # fit copula to data
                copula = pv.Bicop(data = pseudo_merged_data, controls = controls)

                copula_data_final['symbol'].append(symbol)
                copula_data_final['year'].append(year)
                copula_data_final['family'].append(copula.family)
                copula_data_final['rotation'].append(copula.rotation)
                copula_data_final['tau'].append(copula.tau)
                copula_data_final['parameters'].append(copula.parameters)

            year+=1
        print(str(symbol) +  ' done')
        print()

    copula_data_final = pd.DataFrame(copula_data_final)
    copula_data_final = copula_data_final.sort_values(by = ['symbol', 'year'])
    copula_data_final = copula_data_final.reset_index(drop =  True)

    return copula_data_final

def singleStockCheck (stocks, market, symbol, family_set, start_date, end_date):

    controls = pv.FitControlsBicop(family_set = family_set)

    single_stock = stocks[stocks['symbol']== symbol]
    single_stock_year = single_stock[(single_stock['date'] >= start_date) & (single_stock['date'] < end_date)]
    single_market_year = market[(market['date'] >= start_date) & (market['date'] < end_date)]

    merged_data  = pd.merge(single_stock_year, single_market_year, on=['date'], how='inner').drop(columns=['symbol','date'])
    display(merged_data)
    display('tau: ' + str(copula.tau))

    pseudo_merged_data = pv.to_pseudo_obs(merged_data)
    copula = pv.Bicop(data = pseudo_merged_data, controls = controls)
