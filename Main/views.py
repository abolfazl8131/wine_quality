from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import pickle
import pandas as pd
import numpy as np
from sklearn import *
# Create your views here.

class PerdictView(APIView):

    def post(self, request):
        scaler = pickle.load(open('Main/scaler.sav', 'rb'))
        model = pickle.load(open('Main/red_wine_model.sav', 'rb'))
        data = request.data
        df = pd.DataFrame.from_dict(data, orient="index")
        array = df.to_numpy().transpose()
        scaled_data = scaler.transform(array)
        prediction = model.predict(scaled_data)
        status = ''
        if prediction[0] < 4:
            status= 'maybe not so good '
        if prediction[0] > 4 and prediction[0] < 7:
            status = ' avarage side! can be good or bad '
        if prediction[0] > 6:
            status = 'one of the greatest !'

        return Response({'grade':prediction,'advice': status})


