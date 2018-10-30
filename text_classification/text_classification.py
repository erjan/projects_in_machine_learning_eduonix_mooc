import sys
import nltk
import sklearn, pandas as pd, numpy as np
#load the dataset of sms msgs
df = pd.read_table('SMSSpamCollection',header = None,encoding='utf-8')
print(df.info())
print("------------------\n")
print(df.head())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5572 entries, 0 to 5571
    Data columns (total 2 columns):
    0    5572 non-null object
    1    5572 non-null object
    dtypes: object(2)
    memory usage: 43.6+ KB
    None
    ------------------
    
          0                                                  1
    0   ham  Go until jurong point, crazy.. Available only ...
    1   ham                      Ok lar... Joking wif u oni...
    2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
    3   ham  U dun say so early hor... U c already then say...
    4   ham  Nah I don't think he goes to usf, he lives aro...
    
#check class distr-n
classes = df[0]
print(classes.value_counts())


    ham     4825
    spam     747
    Name: 0, dtype: int64
    

## 2. Preprocess the data
#convert class lables to  binary values , 0,1
from sklearn.preprocessing import  LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print(Y[:10])


    [0 0 1 0 0 1 0 0 1 1]
    

 #store the sms msgs data
text_messages = df[1]# 2nd column - the actual smski!
print(text_messages[:10])

    0    Go until jurong point, crazy.. Available only ...
    1                        Ok lar... Joking wif u oni...
    2    Free entry in 2 a wkly comp to win FA Cup fina...
    3    U dun say so early hor... U c already then say...
    4    Nah I don't think he goes to usf, he lives aro...
    5    FreeMsg Hey there darling it's been 3 week's n...
    6    Even my brother is not like to speak with me. ...
    7    As per your request 'Melle Melle (Oru Minnamin...
    8    WINNER!! As a valued network customer you have...
    9    Had your mobile 11 months or more? U R entitle...
    Name: 1, dtype: object
    

#use reg exp to replace emails, urls, phones other numbers, symbols
#replace email adress with 'emailaddr'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr')
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
processed = processed.str.replace(r'£|\$', 'moneysymb')
#replace 10digit phone number with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[s\-]?[\d]{3}[\s-]?[\d]{4}$', 'phonenumber')
#replace normal number with 'number'
processed = processed.str.replace(r'\d+(\.\d+)?', 'number')
processed = processed.str.replace(r'[^\w\d\s]', ' ')
processed = processed.str.replace(r'\s+', ' ')
processed = processed.str.replace(r'^\s+|\s+?$', '')
processed = processed.str.lower()
print(processed)

    0       go until jurong point crazy available only in ...
    1                                 ok lar joking wif u oni
    2       free entry in number a wkly comp to win fa cup...
    3             u dun say so early hor u c already then say
    4       nah i don t think he goes to usf he lives arou...
    5       freemsg hey there darling it s been number wee...
    6       even my brother is not like to speak with me t...
    7       as per your request melle melle oru minnaminun...
    8       winner as a valued network customer you have b...
    9       had your mobile number months or more u r enti...
    10      i m gonna be home soon and i don t want to tal...
    11      six chances to win cash from number to number ...
    12      urgent you have won a number week free members...
    13      i ve been searching for the right words to tha...
    14                      i have a date on sunday with will
    15      xxxmobilemovieclub to use your credit click th...
    16                                 oh k i m watching here
    17      eh u remember how number spell his name yes i ...
    18      fine if that s the way u feel that s the way i...
    19      england v macedonia dont miss the goals team n...
    20               is that seriously how you spell his name
    21      i m going to try for number months ha ha only ...
    22           so pay first lar then when is da stock comin
    23      aft i finish my lunch then i go str down lor a...
    24      ffffffffff alright no way i can meet up with y...
    25      just forced myself to eat a slice i m really n...
    26                          lol your always so convincing
    27      did you catch the bus are you frying an egg di...
    28      i m back amp we re packing the car now i ll le...
    29      ahhh work i vaguely remember that what does it...
                                  ...                        
    5542             armand says get your ass over to epsilon
    5543                u still havent got urself a jacket ah
    5544    i m taking derek amp taylor to walmart if i m ...
    5545        hi its in durban are you still on this number
    5546             ic there are a lotta childporn cars then
    5547    had your contract mobile number mnths latest m...
    5548                     no i was trying it all weekend v
    5549    you know wot people wear t shirts jumpers hat ...
    5550            cool what time you think you can get here
    5551    wen did you get so spiritual and deep that s g...
    5552    have a safe trip to nigeria wish you happiness...
    5553                           hahaha use your brain dear
    5554    well keep in mind i ve only got enough gas for...
    5555    yeh indians was nice tho it did kane me off a ...
    5556    yes i have so that s why u texted pshew missin...
    5557    no i meant the calculation is the same that lt...
    5558                                sorry i ll call later
    5559    if you aren t here in the next lt gt hours imm...
    5560                      anything lor juz both of us lor
    5561    get me out of this dump heap my mom decided to...
    5562    ok lor sony ericsson salesman i ask shuhui the...
    5563                              ard number like dat lor
    5564    why don t you wait til at least wednesday to s...
    5565                                            huh y lei
    5566    reminder from onumber to get number pounds fre...
    5567    this is the numbernd time we have tried number...
    5568                    will b going to esplanade fr home
    5569    pity was in mood for that so any other suggest...
    5570    the guy did some bitching but i acted like i d...
    5571                            rofl its true to its name
    Name: 1, Length: 5572, dtype: object
    

#remove the stopwords from txt msgs
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in sw))
print(processed)

    0       go jurong point crazy available bugis n great ...
    1                                 ok lar joking wif u oni
    2       free entry number wkly comp win fa cup final t...
    3                     u dun say early hor u c already say
    4                  nah think goes usf lives around though
    5       freemsg hey darling number week word back like...
    6          even brother like speak treat like aids patent
    7       per request melle melle oru minnaminunginte nu...
    8       winner valued network customer selected receiv...
    9       mobile number months u r entitled update lates...
    10      gonna home soon want talk stuff anymore tonigh...
    11      six chances win cash number number number poun...
    12      urgent number week free membership number numb...
    13      searching right words thank breather promise w...
    14                                            date sunday
    15      xxxmobilemovieclub use credit click wap link n...
    16                                          oh k watching
    17      eh u remember number spell name yes v naughty ...
    18                             fine way u feel way gota b
    19      england v macedonia dont miss goals team news ...
    20                                   seriously spell name
    21                   going try number months ha ha joking
    22                           pay first lar da stock comin
    23      aft finish lunch go str lor ard number smth lo...
    24                     ffffffffff alright way meet sooner
    25      forced eat slice really hungry tho sucks mark ...
    26                                  lol always convincing
    27      catch bus frying egg make tea eating mom left ...
    28                     back amp packing car let know room
    29               ahhh work vaguely remember feel like lol
                                  ...                        
    5542                          armand says get ass epsilon
    5543                  u still havent got urself jacket ah
    5544    taking derek amp taylor walmart back time done...
    5545                               hi durban still number
    5546                              ic lotta childporn cars
    5547    contract mobile number mnths latest motorola n...
    5548                                     trying weekend v
    5549    know wot people wear shirts jumpers hat belt k...
    5550                                  cool time think get
    5551                         wen get spiritual deep great
    5552    safe trip nigeria wish happiness soon company ...
    5553                                hahaha use brain dear
    5554    well keep mind got enough gas one round trip b...
    5555    yeh indians nice tho kane bit shud go number d...
    5556                      yes u texted pshew missing much
    5557    meant calculation lt gt units lt gt school rea...
    5558                                     sorry call later
    5559                      next lt gt hours imma flip shit
    5560                              anything lor juz us lor
    5561          get dump heap mom decided come lowes boring
    5562    ok lor sony ericsson salesman ask shuhui say q...
    5563                              ard number like dat lor
    5564                     wait til least wednesday see get
    5565                                              huh lei
    5566    reminder onumber get number pounds free call c...
    5567    numbernd time tried number contact u u number ...
    5568                            b going esplanade fr home
    5569                                pity mood suggestions
    5570    guy bitching acted like interested buying some...
    5571                                       rofl true name
    Name: 1, Length: 5572, dtype: object
    
#remove word stems using porter stemmer
ps = nltk.PorterStemmer()
processed =  processed.apply(lambda x: ' '.join(ps.stem(term) for  term in x.split()))
print(processed)

    0       go jurong point crazi avail bugi n great world...
    1                                   ok lar joke wif u oni
    2       free entri number wkli comp win fa cup final t...
    3                     u dun say earli hor u c alreadi say
    4                    nah think goe usf live around though
    5       freemsg hey darl number week word back like fu...
    6           even brother like speak treat like aid patent
    7       per request mell mell oru minnaminungint nurun...
    8       winner valu network custom select receivea num...
    9       mobil number month u r entitl updat latest col...
    10      gonna home soon want talk stuff anymor tonight...
    11      six chanc win cash number number number pound ...
    12      urgent number week free membership number numb...
    13      search right word thank breather promis wont t...
    14                                            date sunday
    15      xxxmobilemovieclub use credit click wap link n...
    16                                             oh k watch
    17      eh u rememb number spell name ye v naughti mak...
    18                             fine way u feel way gota b
    19      england v macedonia dont miss goal team news t...
    20                                     serious spell name
    21                         go tri number month ha ha joke
    22                           pay first lar da stock comin
    23      aft finish lunch go str lor ard number smth lo...
    24                     ffffffffff alright way meet sooner
    25      forc eat slice realli hungri tho suck mark get...
    26                                      lol alway convinc
    27      catch bu fri egg make tea eat mom left dinner ...
    28                        back amp pack car let know room
    29                    ahhh work vagu rememb feel like lol
                                  ...                        
    5542                           armand say get ass epsilon
    5543                  u still havent got urself jacket ah
    5544    take derek amp taylor walmart back time done l...
    5545                               hi durban still number
    5546                               ic lotta childporn car
    5547    contract mobil number mnth latest motorola nok...
    5548                                        tri weekend v
    5549    know wot peopl wear shirt jumper hat belt know...
    5550                                  cool time think get
    5551                           wen get spiritu deep great
    5552    safe trip nigeria wish happi soon compani shar...
    5553                                hahaha use brain dear
    5554    well keep mind got enough ga one round trip ba...
    5555    yeh indian nice tho kane bit shud go number dr...
    5556                            ye u text pshew miss much
    5557    meant calcul lt gt unit lt gt school realli ex...
    5558                                     sorri call later
    5559                       next lt gt hour imma flip shit
    5560                                 anyth lor juz us lor
    5561                get dump heap mom decid come low bore
    5562    ok lor soni ericsson salesman ask shuhui say q...
    5563                              ard number like dat lor
    5564                     wait til least wednesday see get
    5565                                              huh lei
    5566    remind onumb get number pound free call credit...
    5567    numbernd time tri number contact u u number po...
    5568                                b go esplanad fr home
    5569                                    piti mood suggest
    5570    guy bitch act like interest buy someth els nex...
    5571                                       rofl true name
    Name: 1, Length: 5572, dtype: object
    

from nltk.tokenize import word_tokenize
#ccreating a bag of words model
all_words = []
for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)
print("Number of words:{}".format(len(all_words)))
print('most common words:{}'.format(all_words.most_common(15)))


    Number of words:6557
    most common words:[(u'number', 3072), (u'u', 1207), (u'call', 679), (u'go', 456), (u'get', 451), (u'ur', 391), (u'gt', 318), (u'lt', 316), (u'come', 304), (u'ok', 293), (u'free', 284), (u'day', 276), (u'know', 275), (u'love', 266), (u'like', 261)]

#use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]
#define a find_feature function
def find_feature(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word]=(word in words)
    return features
#see results
features = find_feature(processed[0])
for key,value in features.items():
    if value == True:
        print key

    avail
    buffet
    world
    great

processed[0]


    u'go jurong point crazi avail bugi n great world la e buffet cine got amor wat'

#find features for all messages
messages = zip(processed, Y)
#define a  seed for reprod-y
seed = 1
np.random.seed = seed
np.random.shuffle(messages)
#call  find_features function for each sms mesgs
featuresets = [(find_feature(text), label) for (text,label) in messages]

from sklearn import model_selection
training ,testing = model_selection.train_test_split(featuresets,test_size = 0.25,random_state =seed)
print('training:{}'.format(len(training)))
print('testing:{}'.format(len(testing)))


    training:4179
    testing:1393
    

## 4.scikitlearn classfiers with NLTK

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression, SGDClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.svm import  SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

#define models to train
names = ['K nearest neighbors', 'Decision tree', 'Random forest', 'logistic regression', 'SGD classifier', 'Naive bayes', 'SVM linear']
classifiers = [
    KNeighborsClassifier(), 
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
]
models = zip(names, classifiers)

print(models)

    [('K nearest neighbors', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')), ('Decision tree', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')), ('Random forest', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)), ('logistic regression', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)), ('SGD classifier', SGDClassifier(alpha=0.0001, average=False, class_weight=None,
           early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
           l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=100,
           n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
           power_t=0.5, random_state=None, shuffle=True, tol=None,
           validation_fraction=0.1, verbose=0, warm_start=False)), ('Naive bayes', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)), ('SVM linear', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))]
#wrap models in NLTK
from nltk.classify.scikitlearn import  SklearnClassifier
for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing)*100 
    print('{}:accuracy:{}'.format(name,accuracy))
    
    K nearest neighbors:accuracy:94.472361809
    Decision tree:accuracy:95.7645369706
    Random forest:accuracy:96.2670495334
    logistic regression:accuracy:96.7695620962
    SGD classifier:accuracy:96.5541995693
    Naive bayes:accuracy:95.9798994975
    SVM linear:accuracy:96.5541995693

#ensemble method  voting classifier
from sklearn.ensemble import  VotingClassifier
#define models to train
from nltk.classify.scikitlearn import  SklearnClassifier
for name,model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model,testing)*100 
    #print('{}:accuracy:{}'.format(name,accuracy))

models = zip(names,classifiers)
nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models,voting='hard', n_jobs= -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_ensemble,testing)*100
print('ensemble method accuracy :{}'.format(accuracy))

    ensemble method accuracy :96.9131371141

#make class label prediction foro testing set
txt_features,labels = zip(*testing)
prediction = nltk_ensemble.classify_many(txt_features)

#print a confusion matrix & a classification report
print(classification_report(labels,prediction))
pd.DataFrame(
confusion_matrix(labels,prediction),
index = [['actual','actual'],['ham','spam']],
columns = [['predicted','predicted'], ['ham','spam']]
)


                  precision    recall  f1-score   support
    
               0       0.97      1.00      0.98      1213
               1       0.98      0.78      0.87       180
    
       micro avg       0.97      0.97      0.97      1393
       macro avg       0.97      0.89      0.92      1393
    weighted avg       0.97      0.97      0.97      1393    
