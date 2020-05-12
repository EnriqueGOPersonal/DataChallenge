# -*- coding: utf-8 -*-
"""
Created on Mon May 11 03:43:13 2020

@author: enriq
"""

import urllib, os

myloc = r"C:\Users\enriq\Documents\GitHub\DataChallenge\data" #replace with your own location
key = "&key=" + "" #got banned after ~100 requests with no key

def GetStreet(Add,SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + '"' + Add + '"' + key #added url encoding
  fi = Add + ".jpg"
  urllib.request.urlopen(MyUrl, os.path.join(SaveLoc,fi))

Tests = ["457 West Robinwood Street, Detroit, Michigan 48203",
         "1520 West Philadelphia, Detroit, Michigan 48206",
         "2292 Grand, Detroit, Michigan 48238",
         "15414 Wabash Street, Detroit, Michigan 48238",
         "15867 Log Cabin, Detroit, Michigan 48238",
         "3317 Cody Street, Detroit, Michigan 48212",
         "14214 Arlington Street, Detroit, Michigan 48212"]

for i in Tests:
  GetStreet(Add=i,SaveLoc=myloc)