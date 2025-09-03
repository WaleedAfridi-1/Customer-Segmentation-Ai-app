from django.shortcuts import render
import pickle
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "segmentation", "static", "kmeans_model.pkl")

with open(model_path, "rb") as file:
    kmeans_model = pickle.load(file)


def home(request):
    cluster = None
    age = ''
    income = ''
    score = ''
    
    if request.method == "POST":
        age = int(request.POST.get("age"))
        income = int(request.POST.get("income"))  
        score = int(request.POST.get("score"))

        
        cluster = kmeans_model.predict([[age, income, score]])[0]

    return render(request, "segmentation/home.html", {
        "cluster": cluster,
        "age": age,
        "income": income,
        "score": score
    })