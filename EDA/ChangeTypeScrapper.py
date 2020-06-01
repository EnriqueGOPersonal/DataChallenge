
import pandas as pd
from bs4 import BeautifulSoup
import requests 
import re
from datetime import datetime, timedelta
import locale
locale.setlocale(locale.LC_ALL,'en_US.UTF-8')

def ScrapChangeTypes(min_date, now = datetime.now()):
    months = {
        "January" : "enero", 
        "February" : "febrero", 
        "March" : "marzo", 
        "April": "abril", 
        "May": "mayo", 
        "June": "junio", 
        "July": "julio", 
        "August": "agosto", 
        "September": "septiembre", 
        "October": "octubre", 
        "November": "noviembre", 
        "December": "diciembre"
    }
    
    n_months = {
        "enero": "01", 
        "febrero": "02", 
        "marzo": "03", 
        "abril": "04", 
        "mayo": "05", 
        "junio": "06", 
        "julio": "07", 
        "agosto": "08", 
        "septiembre": "09", 
        "octubre": "10", 
        "noviembre": "11", 
        "diciembre": "12"
    }
    
    start = min_date
    end = now
    
    dates = []
    for d in range((end - start).days):
        date = (start + timedelta(d)).strftime(r"%B-%Y")
        date = date.split("-")
        date[0] = months[date[0]]
        date = '-'.join(date)
        if date not in dates:
            dates.append(date)
                
    hist = []
    
    for date in dates:
        url = "https://cambio.today/historico/sol/dolar-norteamericano/" + date
        r = requests.get(url)
        soup = BeautifulSoup(r.text)
        for i in soup.find_all("td"):
            print(i.text)
            if "PEN USD" not in i.text:
                hist.append(i.text)
    
    hist = pd.Series(hist)
    
    df_dict = {"date": hist.loc[hist.index.values%2 == 0].reset_index(drop = True),
               "dollar_change": hist[hist.index%2 != 0].reset_index(drop = True)}
     
    df = pd.DataFrame(df_dict)
    
    df.dollar_change = df.dollar_change.apply(lambda x: re.sub(".*= ", "", x))
    df.dollar_change = df.dollar_change.apply(lambda x: re.sub(" .*", "", x))
    df.dollar_change = df.dollar_change.apply(lambda x: float(re.sub(",", ".", x)))
    
    df.date = df.date.apply(lambda x: re.sub(".*, ", "", x))
    df.date = df.date.apply(lambda x: re.sub(" de ", "/", x))
    df.date = df.date.apply(lambda x: '/'.join([x.split("/")[0], n_months[x.split("/")[1]], x.split("/")[2]]))
    
    df["month"] = df.date.apply(lambda x: x.split("/")[1])
    
    df.date = pd.to_datetime(df.date, format = '%d/%m/%Y')
    
    df.to_csv("data/change_type_" + str(now.date()) + ".csv")

    return df

a = ScrapChangeTypes()



