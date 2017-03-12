#Formulae Used in Alpha

###Overnight Return, $ris$
$$ris = \ln{(\frac{Open_{s,d}}{Close_{s,d-1}})}$$

###Intercept, $itc$ or $\beta_{s,d}^{0}$
$$itc = 1$$

###Size, $prc$ or $\beta_{s,d}^{1}$
$$prc = \ln{(Close_{s,d-1})}$$

###Momentum, $mom$ or $\beta_{s,d}^{2}$
$$mom = \ln{(\frac{Close_{s,d-1}}{Open_{s,d-1}})}$$

###Intraday Volatility, $hlv$ or $\beta_{s,d}^{3}$
$$hlv = \ln{\sqrt{\frac{1}{21}\sum_{k=1}^{21}{(\frac{High_{s,d-k}-Low_{s,d-k}}{Close_{s,d-k}})}^{2}}}$$
<center>**OR**</center>
$$hlv = \ln{\sqrt{SMA_{21}{(\frac{High_{s,d-k}-Low_{s,d-k}}{Close_{s,d-k}})}^{2}}}$$
$SMA_{21}$ = Simple Moving Average of 21

###Volume Moving Average, $vol$ or $\beta_{s,d}^{4}$
$$vol = \ln{(\frac{1}{21}\sum_{k=1}^{21}Close_{s,d-k})}$$
<center>**OR**</center>
$$vol = \ln{(SMA_{21}(Close_{s,d-k}))}$$

###Liquidity, $liq$
$$liq = Close_{s,d-1} \times Volume_{s,d-1}$$

###Intraday Return, $ir$
$$ir = \frac{Close_{s,d}}{Open_{s,d}}-1$$

###Notes: 
Both $hlv$ and $vol$ are normalized by subtracting their mean values.
Residual and Weight variables are to be implemented.