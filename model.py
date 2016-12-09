
## 
##
##  Implementation of 
##  4-Factor Model for Overnight Returns  
##
##  Version 2
##

import sys
import os
import datetime
import math
import csv
import statistics
import numpy as np
from sklearn import linear_model

class Model:

	def __init__(self):

		# Regression Parameters
		self.file_limit = 10000   # Maximum number of files to read 
		self.min_quotes = 20    # Only ticker with greater than this will be used
		self.volatility_days = self.min_quotes -1 # Days used to compute volatility
		self.sampling_count = 2000 # Select x stock files by liquidity for regression each day.

		# Backtesting Parameters
		self.investment = 10000000       # Initial investment
		self.portfolio_size = 50         # Maximum number of stocks in the portfolio
		self.fee_value = 0.0000218       # Transaction fee based on value sold
		self.fee_quantity = 0.000119     # Transaction fee based on quantity sold
		self.fee_min = 1                 # Minimum transaction fee per transaction (applied on selling only)
		#self.risk_free_rate = 0.05       # Any unused funds will gain risk-free rate. (Not implemented)
		#self.capital_gain_tax = 0        # Tax rate applied on profits (Not implemented)
		self.spread = 0.01                # Spread expressed as absolute value
		self.allow_short = True          # Shortselling stocks is allowed
		#self.margin_requirement = 1      # Margin requirement for shortselling expressed as % (Not implemented) 
		self.divisible_stocks = False     # Allow shares to be infinitely divisible

		# Variables
		self._cwd = os.path.dirname(os.path.realpath(__file__))  # Current working directory
		self._files = []  # Store name of quotes files
		self._data = {} # Store parsed quote files (grouped by ticker)
		self._daily_quotes = {} # Store processed quote files (grouped by date)
		self._coef = {} # Store daily factor coefficients (grouped by date)
		self._reference_gain = {} # Store reference index gain (simple average over return for each day)
		self._daily_coeffs = {}
		self._daily_stats = {}

		# Run
		self.scan_folder()
		self.load_data()
		self.process_data()
		self.normalise_factors()
		self.select_samples()
		self.compute_coeff()
		self.write_results_to_file()
		self.backtest()

	def scan_folder(self):
		'''
		Scan the _cwd folder and record all csv files inside into _files.
		limit: maximum number of files to read.
		'''

		for file in os.listdir(self._cwd+"/quotes/"):
			if file.endswith(".csv"):
				self._files.append(file)
			if len(self._files) >= self.file_limit:
				break
		print("%d files found." % (len(self._files)))


	def load_data(self):
		'''
		Load the files listed in _files into _data.
		Generate a new key for each stock.
		Store stock data as a list of tuple in ascending order of date.
		(0 Ticker, 1 Date, 2 Open, 3 High, 4 Low, 5 Close, 6 Volume, 7 Adj Close)
		'''

		quotes_count = 0
		counter = 0
		discarded = []

		for file in self._files:
			with open(self._cwd+"/quotes/"+file) as f:
				reader = csv.reader(f)
				ticker = file.strip(".csv")
				quotes = []
				for row in reader:
					if row[0] == "Date":
						continue
					date = row[0].split("-")
					date = datetime.date(int(date[0]), int(date[1]), int(date[2]))
					# (0 Ticker, 1 Date, 2 Open, 3 High, 4 Low, 5 Close, 6 Volume, 7 Adj Close)
					quotes.append((ticker, date, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])))
					if date<datetime.date(2010,1,1):
						continue
				quotes.sort(key=lambda x: x[1]) # Sort the quotes by date
				# Discard any ticker with less than required quotes
				if len(quotes) > self.min_quotes:
					self._data[ticker] = quotes
					quotes_count += len(quotes)
				else:
					discarded.append(ticker)
			
			counter += 1
			print("Parsing raw data ... %d / %d (%1.0f%%)"%(counter, len(self._files), 100*counter/len(self._files)), end="\r")

		sys.stdout.write("\n")
		print("%d files loaded into memory (%d quotes recorded)." % (len(self._data), quotes_count))
		print("%d files discarded for having too few data: " % (len(discarded)), end="")
		print(discarded)



	def process_data(self):
		'''
		Compute the required fields.
		Record the required fields into _coef.
		Create a new key for each day.
		Store stock data as a list of tuple.
		(0 Name, 
		 1 Overnight Return "Ris", 
		 2 Intercept "itc", 
		 3 Price "prc", 
		 4 Momentum "mom", 
		 5 Intraday Votality "hlv", 
		 6 Volume "vol"
		 7 Liquidity "liq"
		 8 Intraday_return "ir"
		 9 Dividend payout, "div"
		 10 Open, close
		 11 Residual -> to be implemented later
		 12 Weight -> to be implemented later) 
		'''
		#(0 Name, 1 Date, 2 Open, 3 High, 4 Low, 5 Close, 6 Volume, 7 Adj Close)

		counter = 0
		temp_vol = 0 # To record last non-zero vol factor

		for stock in self._data:
			d = self._data[stock]
			for i in range(1+self.volatility_days, len(d)):
				# Compute factors
				# Overnight return
				ris = math.log(d[i][2]/d[i-1][5])
				# Intercept
				itc = 1
				# Size
				prc = math.log(d[i-1][5])
				# Momentum
				mom = math.log(d[i-1][5]/d[i-1][2])
				# Intraday Votality
				uis = sum(((d[i-j][3]-d[i-j][4])/d[i-j][5])**2 for j in range(1, self.volatility_days+2))/(self.volatility_days+1)
				if uis != 0:
					hlv = 0.5*math.log(uis)
				else: 
					hlv = 0
				# Volume
				vis = sum(d[i-j][6] for j in range(1, self.volatility_days+2))/(self.volatility_days+1)
				if vis != 0:
					vol = math.log(vis) 
					temp_vol = vol
				else:
					vol = temp_vol   # If volume is zero, use the last non-zero vol factor.
				# Liquidity
				liq = d[i-1][5]*d[i-1][6]
				# Intraday return
				ir = d[i][5]/d[i][2]-1
				# Dividend payout
				div = (d[i][7]/d[i-1][7])/(d[i][5]/d[i-1][5]) - 1
				if (div < 0.00001) or (div > 0.15):
					div = 0
				# Record factors into _daily_quotes
				rec = (d[0][0], ris, itc, prc, mom, hlv, vol, liq, ir, div, (d[i][2], d[i][5]))
				if d[i][1] in self._daily_quotes:
					self._daily_quotes[d[i][1]].append(rec)
				else:
					self._daily_quotes[d[i][1]] = [rec,]

			counter += 1
			print("Processing data ... %d / %d (%1.0f%%)"%(counter, len(self._data), 100*counter/len(self._data)), end="\r")

		sys.stdout.write("\n")
		print("%d stocks data processed." % (len(self._data)))



	def normalise_factors(self):
		'''
		hlv and vol will be normalized.
		'''
		counter = 0

		for day in self._daily_quotes:
			hlv, vol = [], []
			for quote in self._daily_quotes[day]:
				hlv.append(quote[5])
				vol.append(quote[6])

			hlv_mean = statistics.mean(hlv)
			vol_mean = statistics.mean(vol)

			for i in range(len(self._daily_quotes[day])):
				data = self._daily_quotes[day][i]
				r_hlv = data[5]-hlv_mean
				r_vol = data[6]-vol_mean
				self._daily_quotes[day][i] = (data[0], data[1], data[2], data[3], data[4], r_hlv, r_vol, data[7], data[8], data[9], data[10])

			self._reference_gain[day] = sum(x[8] for x in self._daily_quotes[day])/len(self._daily_quotes[day])

			counter += 1
			print("Normalising factors ... %d / %d (%1.0f%%)"%(counter, len(self._daily_quotes), 100*counter/len(self._daily_quotes)), end="\r")

		sys.stdout.write("\n")
		print("Normalised %d days of data." % (len(self._daily_quotes)))



	def select_samples(self):
		'''
		For each day, keep only n samples based on liquidity in _daily_quotes.
		'''
		for day in self._daily_quotes:
			if len(self._daily_quotes[day]) > self.sampling_count:
				self._daily_quotes[day].sort(key=lambda x: x[7], reverse=True)
				self._daily_quotes[day] = self._daily_quotes[day][:self.sampling_count]
		print("Completed sampling of data.")



	def compute_coeff(self):
		'''
		Compute coefficients using linear regression.
		0 itc
		1 prc
		2 mom
		3 hlv
		4 vol
		'''
		counter = 0

		for day in self._daily_quotes:
			clf = linear_model.LinearRegression()
			data = self._daily_quotes[day]

			y = list(x[1] for x in data)
			x = list([x[2], x[3], x[4], x[5], x[6]] for x in data)

			clf.fit(x, y)

			coeffs = clf.coef_.copy()
			coeffs[0] = clf.intercept_
			self._daily_coeffs[day] = coeffs
			self._daily_stats[day] = {'score': clf.score(x, y), 'ssize':len(data)}

			counter += 1
			print("Computing coefficients ... %d / %d (%1.0f%%)"%(counter, len(self._daily_quotes), 100*counter/len(self._daily_quotes)), end="\r")



	def write_results_to_file(self):
		print("\nWriting coefficients to file...")

		entries = []
		for day in self._daily_coeffs:
			entry = [day,]
			entry.extend(self._daily_coeffs[day])
			entry.append(self._daily_stats[day]['score'])
			entry.append(self._daily_stats[day]['ssize'])
			entries.append(entry)
		entries.sort(key=lambda x:x[0])

		with open("coeffs.txt", 'w') as f:
			f.write("Date,int,prc,mom,hlv,vol,R^2,s_size\n")
			for entry in entries:
				f.write("%s,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%1.8f,%d\n"%(entry[0].isoformat(), entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7]))

		print("Done!")



	def backtest(self):
		print("\nTesting results...")
		counter = 0
		# Variables
		daily_weight_multiplier = {}
		daily_t_stats = {}
		daily_stats = [] # (Day, Value, Gain, Fee, BeginVal)
		daily_holdings = {}
		prev_value = self.investment

		# Compute weights
		days = list(self._daily_quotes.keys())
		days.sort()
		for day in days:
			coeffs = self._daily_coeffs[day]
			data = self._daily_quotes[day]
			sum_residuals, sum_epsilon2 = 0, 0

			# Finding weight multiplier
			for i in range(len(data)):
				# resid = Ris    -itc      -prc              -mom              -hlv              -vol
				epsilon = data[i][1]-coeffs[0]-data[i][3]*coeffs[1]-data[i][4]*coeffs[2]-data[i][5]*coeffs[3]-data[i][6]*coeffs[4]
				sum_residuals += abs(epsilon)
				sum_epsilon2 += epsilon**2
				data[i] = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9], data[i][10], epsilon)

			if sum_residuals != 0:
				weight_multiplier = -1/sum_residuals
				daily_weight_multiplier[day] = weight_multiplier
			else:
				weight_multiplier = 0
				continue

			# Compute dollar holdings for each stock
			temp_holdings = []
			for i in range(len(data)):
				weight = weight_multiplier * data[i][11]
				if data[i][9] > 0:
					weight = 0
				data[i] = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], data[i][8], data[i][9], data[i][10], data[i][11], weight)
				if self.allow_short:
					temp_holdings.append((i, weight, abs(weight)))
				else:
					temp_holdings.append((i, 0, 0))

			# Recalibrate weight
			temp_holdings.sort(key=lambda x: x[2], reverse=True)
			temp_holdings = temp_holdings[:self.portfolio_size]
			sum_weight = sum(x[2] for x in temp_holdings)
			holdings = []
			for x in temp_holdings:
				# (index, ticker, shares)
				shares = prev_value * (x[1]/sum_weight) / data[x[0]][10][0]
				if not self.divisible_stocks:
					shares = int(shares)
				holdings.append((x[0], data[x[0]][0], shares))
			daily_holdings[day] = holdings

			# Compute gains
			# (index, ticker, shares)
			fee, gain = 0, 0
			for x in holdings:
				fee += max(self.fee_min, self.fee_quantity * abs(x[2]) + self.fee_value * abs(x[2]) * data[x[0]][10][0])
				gain += (data[x[0]][10][1] - data[x[0]][10][0] - self.spread) * x[2] # (C-O-spread)*shares

			# Compute T-stats
			t_hat = (sum_epsilon2/len(self._data))**0.5
			n_stocks = len(self._daily_quotes[day])
			# Ensure no singular matrix
			if n_stocks >= 0.1*self.sampling_count:
				x = np.array(list([x[2], x[3], x[4], x[5], x[6]] for x in self._daily_quotes[day]))
				x_t = np.transpose(x)
				product = np.dot(x_t, x)
				temp = np.linalg.inv(product)
				lam = list(temp[i][i]*t_hat for i in range(5)) # diagonal entries
				t_stats = list(coeffs[i]/lam[i] for i in range(5))
				daily_t_stats[day] = t_stats

			# Record daily stats
			print("%s\t%1.0f"%(day.isoformat(), prev_value))
			daily_stats.append((day, prev_value + gain - fee, gain, fee, prev_value))
			prev_value = prev_value + gain - fee

			counter += 1
			print("Backtesting ... %d / %d (%1.0f%%)"%(counter, len(self._daily_quotes), 100*counter/len(self._daily_quotes)), end="\r")

		# Print result to file
		with open("positions.txt",'w') as f:
			for i in range(len(daily_stats)):
				day = daily_stats[i][0]
				value = 0
				for x in daily_holdings[day]:
					value += abs(x[2]) * self._daily_quotes[day][x[0]][10][0]
				f.write("%s\t%1.0f\t%1.0f\t%s\n"%(day.isoformat(), value, daily_stats[i][4], str(daily_holdings[day])))

		sys.stdout.write("\n")

		# Compute final result
		daily_stats.sort(key=lambda x:x[0]) # (Day, Value, Gain, Fee, prev_val)
		result = {}
		result['total days'] = len(daily_stats)
		result['winning days'] = sum(1 for x in daily_stats if x[1]>=x[4])
		result['losing days'] = sum(1 for x in daily_stats if x[1]<x[4])
		result['win rate'] = result['winning days']/result['total days']
		result['average win'] = sum(x[1]/x[4]-1 for x in daily_stats if x[1]>=x[4])/result['winning days']
		result['average loss'] = sum(-x[1]/x[4]-1 for x in daily_stats if x[1]<x[4])/result['losing days']
		result['profit factor'] = sum(x[1]/x[4]-1 for x in daily_stats if x[1]>=x[4])/sum(-x[1]/x[4]-1 for x in daily_stats if x[1]<x[4])
		
		result['cumulative return'] = daily_stats[-1][1]/self.investment - 1
		result['average daily return'] = (1+result["cumulative return"])**(1/result["total days"]) - 1
		result['max daily return'] = max(x[1]/x[4]-1 for x in daily_stats)
		result['max daily drawdown'] = min(x[1]/x[4]-1 for x in daily_stats)

		result['daily return stdev'] = statistics.stdev(list(x[1]/x[4]-1 for x in daily_stats))
		result['information ratio'] = result['average daily return']/result['daily return stdev']
		result['sharpe ratio'] = result['information ratio']*(252**0.5)

		result['total transaction fee'] = sum(x[3] for x in daily_stats)
		
		result['reference cumulative return'] = 1
		for key in self._reference_gain:
			result['reference cumulative return'] = result['reference cumulative return'] * (1+self._reference_gain[key])
		result['reference cumulative return'] -= 1
		
		result['max consecutive wins'] = 0
		result['max consecutive losses'] = 0
		result['max drawdown'] = 1

		record = []
		current_ref_value = self.investment
		wins_count, losses_count, drawdown_count = 0, 0, 1

		for i in range(len(daily_stats)): # (Day, Value, Gain, Fee, prev_val)
			current_ref_value += current_ref_value * self._reference_gain[daily_stats[i][0]]
			# (Day, Gain%, Value, Ref_value, daily_fee)
			record.append((daily_stats[i][0], daily_stats[i][1]/daily_stats[i][4]-1, daily_stats[i][1], current_ref_value, daily_stats[i][3]))
			if daily_stats[i][1]>daily_stats[i][4]:
				wins_count += 1
				losses_count = 0
				drawdown_count = 1
				if wins_count > result['max consecutive wins']:
					result['max consecutive wins'] = wins_count
			elif daily_stats[i][1]<daily_stats[i][4]:
				wins_count = 0
				losses_count += 1
				drawdown_count *= (daily_stats[i][1]/daily_stats[i][4])
				if losses_count > result['max consecutive losses']:
					result['max consecutive losses'] = losses_count
				if drawdown_count < result['max drawdown']:
					result['max drawdown'] = drawdown_count

		result['max drawdown'] = 1 - result['max drawdown']

		result['average_t_stats'] = [0,0,0,0,0]
		for day in daily_t_stats:
			for i in range(5):
				result['average_t_stats'][i] += abs(daily_t_stats[day][i])
		for i in range(5):
			result['average_t_stats'][i] = result['average_t_stats'][i]/result['total days']

		self._results = result

		# Print result to file
		with open("result_new.txt",'w') as f:
			for key in result:
				f.write("%s = %s\n"%(key, str(result[key])))

			f.write("\nDate\tGain%\tVal\tRef_value\tDaily_fee\n")

			for i in range(len(daily_stats)):
				f.write("%s\t%1.2f%%\t%1.0f\t%1.0f\t%1.0f\n"%(daily_stats[i][0].isoformat(), record[i][1]*100, record[i][2], record[i][3], record[i][4]))

		print("Done!")



if __name__ == "__main__":
	Model()
	input("Press any key to continue")

