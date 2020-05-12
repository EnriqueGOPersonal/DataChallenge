
import pandas as pd
from bs4 import BeautifulSoup
import requests 
import re
from datetime import datetime

def ScrapChangeTypes(now = datetime.now()):

    years = range(2017, now.year+1)
    
    months = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    converter = pd.DataFrame(months, columns = ["month"])
    converter["month_number"] = range(1,13)
    converter["month_number"] = converter["month_number"].apply(lambda x: str(x).zfill(2))
    
    hist = []
    
    for m in months:
        for y in years:
            url = "https://cambio.today/historico/sol/dolar-norteamericano/" + m +"-" + str(y)
            r = requests.get(url)    
            soup = BeautifulSoup(r.text)
            for i in soup.find_all("td"):
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
    
    df["month"] = df.date.apply(lambda x: x.split("/")[1])
    
    df = df.merge(converter, on = "month")
    df.to_csv("data/change_type_" + str(now.date()) + ".csv")

    return df

a = ScrapChangeTypes()



