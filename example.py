from joblib import load

# Load the model and vectorizer
model = load('voting_classifier_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Input: A cookie theft transcription string
# Output: A classification of 0 for control or 1 for dementia
def classify_text(input_text):

    # uncased
    cleaned_input = input_text.lower()

    # only words or characters 
    cleaned_input = ''.join([char for char in cleaned_input if char.isalpha() or char.isspace()])

    # convert to tfidf tokens
    input_vectorized = vectorizer.transform([cleaned_input])
    
    # run the model to predict 0 for control or 1 for dementia
    prediction = model.predict(input_vectorized)
    
    return prediction[0]

###Example usage###

control_string = """the scene is in the  in the kitchen
the mother is wiping dishes and the water is running on the
a child is trying to get  a boy is trying to get cookiesoutta  out of a jar and hes about to tip over on a stool
uh the little girl is reacting to his falling
uh it seems to be summer out
the window is open
the curtains are blowing
it must be a gentle breeze
theres grass outside in the garden
uh mothers finished certain of the  the dishes
kitchens very tidy
the mother seems to have nothing in the house to eat except cookiesin the cookie jar
uh the children look to be almost about the same size
perhaps theyre twins
theyre dressed for summer warm weather
um you want more
 the mothers in a short sleeve dress
 ill hafta say its warm"""

classification = classify_text(control_string)
print("The classification is:", classification)

dementia_string = """does that say cookie honey?
oh cookie jar.
this chair [: stool] [* s:r] is tilted.
do I tell you that too?
+< titled chair [: stool] [* s:r].
girl goin(g) [: doin(g)] [* s:r] dishes.
spillin(g) water.
(.) here's two plates.
now what would I say about them?
water spillin(g).
(.) cookie jar.
(.) chair [: stool] [* s:r] fallin(g).
and children.
hell this is a damn dumb thing."""

classification = classify_text(dementia_string)
print("The classification is:", classification)
