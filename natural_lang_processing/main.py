import nltk
import sys
import sklearn

print("python: {}".format(sys.version))
print("nltk version: {}".format(nltk.__version__))
print("sklearn :{}".format(sklearn.__version__))
from nltk.tokenize import  sent_tokenize, word_tokenize
text = "hello students, how are you doing today? the olympics are inspiring and python is awesome, u look great today."
print(sent_tokenize(text))
print( word_tokenize(text))
#removing stop words 
from nltk.corpus import stopwords
print(set(stopwords.words('english'))) 
example = 'this s some sample text, showing off the stop words filtration.'
stop_words = set(stopwords.words('english'))
word_tokens =  word_tokenize(example)
sentence = [ w for  w in word_tokens if not w in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in  stop_words:
        filtered_sentence.append(w)
        
print("word tokens" + str(word_tokens))
print(sentence)
print(filtered_sentence)
#stemming words with NLTK
from nltk.stem import PorterStemmer
ps = PorterStemmer()
example_words = ['ride', 'riding','rider','rides']
for w in example_words:
    print(ps.stem(w))
#stemming the entire sentence
new_text = 'when riders ride their horses, they often think of how cowboys rode horses.'
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))
nltk.download()
from nltk.corpus import udhr
print(udhr.raw("English-Latin1"))
from nltk.corpus import state_union
from nltk.tokenize import  PunktSentenceTokenizer
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
print(train_text)
#now u can train  the punt  sentence tokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
#now tokenize the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)
print(tokenized)
#define a functhion that will  tag each  tokenized word witha  part of speech
def process_content():
    try:
        for i in  tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
process_content()
nltk.help.upenn_tagset()
$: dollar
    $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
'': closing quotation mark
    ' ''
(: opening parenthesis
    ( [ {
): closing parenthesis
    ) ] }
,: comma
    ,
--: dash
    --
.: sentence terminator
    . ! ?
:: colon or ellipsis
    : ; ...
CC: conjunction, coordinating
    & 'n and both but either et for less minus neither nor or plus so
    therefore times v. versus vs. whether yet
CD: numeral, cardinal
    mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-
    seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
    fifteen 271,124 dozen quintillion DM2,000 ...
DT: determiner
    all an another any both del each either every half la many much nary
    neither no some such that the them these this those
EX: existential there
    there
FW: foreign word
    gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
    lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
    terram fiche oui corporis ...
IN: preposition or conjunction, subordinating
    astride among uppon whether out inside pro despite on by throughout
    below within for towards near behind atop around if like until below
    next into if beside ...
JJ: adjective or numeral, ordinal
    third ill-mannered pre-war regrettable oiled calamitous first separable
    ectoplasmic battery-powered participatory fourth still-to-be-named
    multilingual multi-disciplinary ...
JJR: adjective, comparative
    bleaker braver breezier briefer brighter brisker broader bumper busier
    calmer cheaper choosier cleaner clearer closer colder commoner costlier
    cozier creamier crunchier cuter ...
JJS: adjective, superlative
    calmest cheapest choicest classiest cleanest clearest closest commonest
    corniest costliest crassest creepiest crudest cutest darkest deadliest
    dearest deepest densest dinkiest ...
LS: list item marker
    A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
    SP-44007 Second Third Three Two * a b c d first five four one six three
    two
MD: modal auxiliary
    can cannot could couldn't dare may might must need ought shall should
    shouldn't will would
NN: noun, common, singular or mass
    common-carrier cabbage knuckle-duster Casino afghan shed thermostat
    investment slide humour falloff slick wind hyena override subhumanity
    machinist ...
NNP: noun, proper, singular
    Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
    Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
    Shannon A.K.C. Meltex Liverpool ...
NNPS: noun, proper, plural
    Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
    Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
    Apache Apaches Apocrypha ...
NNS: noun, common, plural
    undergraduates scotches bric-a-brac products bodyguards facets coasts
    divestitures storehouses designs clubs fragrances averages
    subjectivists apprehensions muses factory-jobs ...
PDT: pre-determiner
    all both half many quite such sure this
POS: genitive marker
    ' 's
PRP: pronoun, personal
    hers herself him himself hisself it itself me myself one oneself ours
    ourselves ownself self she thee theirs them themselves they thou thy us
PRP$: pronoun, possessive
    her his mine my our ours their thy your
RB: adverb
    occasionally unabatingly maddeningly adventurously professedly
    stirringly prominently technologically magisterially predominately
    swiftly fiscally pitilessly ...
RBR: adverb, comparative
    further gloomier grander graver greater grimmer harder harsher
    healthier heavier higher however larger later leaner lengthier less-
    perfectly lesser lonelier longer louder lower more ...
RBS: adverb, superlative
    best biggest bluntest earliest farthest first furthest hardest
    heartiest highest largest least less most nearest second tightest worst
RP: particle
    aboard about across along apart around aside at away back before behind
    by crop down ever fast for forth from go high i.e. in into just later
    low more off on open out over per pie raising start teeth that through
    under unto up up-pp upon whole with you
SYM: symbol
    % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
TO: "to" as preposition or infinitive marker
    to
UH: interjection
    Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
    huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
    man baby diddle hush sonuvabitch ...
VB: verb, base form
    ask assemble assess assign assume atone attention avoid bake balkanize
    bank begin behold believe bend benefit bevel beware bless boil bomb
    boost brace break bring broil brush build ...
VBD: verb, past tense
    dipped pleaded swiped regummed soaked tidied convened halted registered
    cushioned exacted snubbed strode aimed adopted belied figgered
    speculated wore appreciated contemplated ...
VBG: verb, present participle or gerund
    telegraphing stirring focusing angering judging stalling lactating
    hankerin' alleging veering capping approaching traveling besieging
    encrypting interrupting erasing wincing ...
VBN: verb, past participle
    multihulled dilapidated aerosolized chaired languished panelized used
    experimented flourished imitated reunifed factored condensed sheared
    unsettled primed dubbed desired ...
VBP: verb, present tense, not 3rd person singular
    predominate wrap resort sue twist spill cure lengthen brush terminate
    appear tend stray glisten obtain comprise detest tease attract
    emphasize mold postpone sever return wag ...
VBZ: verb, present tense, 3rd person singular
    bases reconstructs marks mixes displeases seals carps weaves snatches
    slumps stretches authorizes smolders pictures emerges stockpiles
    seduces fizzes uses bolsters slaps speaks pleads ...
WDT: WH-determiner
    that what whatever which whichever
WP: WH-pronoun
    that what whatever whatsoever which who whom whosoever
WP$: WH-pronoun, possessive
    whose
WRB: Wh-adverb
    how however whence whenever where whereby whereever wherein whereof why
``: opening quotation mark
    ` ``
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in  tokenized[:20]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # combine the part of speech tag with a reg exp
            chunkGramm = r"""Chunk:{<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGramm)
            chunked = chunkParser.parse(tagged)
            #print the nltk tree
            for subtree in chunked.subtrees(filter =lambda t: t.label()== 'Chunk'):
                print(subtree)
            #draw the chunks with  nltk
            chunked.draw()
    except Exception as e:
        print(str(e))
process_content()
(Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)
(Chunk ADDRESS/NNP)
(Chunk A/NNP JOINT/NNP SESSION/NNP)
(Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)
(Chunk THE/NNP UNION/NNP January/NNP)
(Chunk THE/NNP PRESIDENT/NNP)
(Chunk Thank/NNP)
(Chunk Mr./NNP Speaker/NNP)
(Chunk Vice/NNP President/NNP Cheney/NNP)
(Chunk Congress/NNP)
(Chunk Supreme/NNP Court/NNP)
(Chunk called/VBD America/NNP)
(Chunk Coretta/NNP Scott/NNP King/NNP)
(Chunk Applause/NNP)
(Chunk President/NNP George/NNP W./NNP Bush/NNP)
(Chunk State/NNP)
(Chunk Union/NNP Address/NNP)
(Chunk Capitol/NNP)
(Chunk Tuesday/NNP)
(Chunk Jan/NNP)
(Chunk White/NNP House/NNP photo/NN)
(Chunk Eric/NNP DraperEvery/NNP time/NN)
(Chunk Capitol/NNP dome/NN)
(Chunk have/VBP served/VBN America/NNP)
(Chunk Tonight/NNP)
(Chunk Union/NNP)
(Chunk Applause/NNP)
(Chunk United/NNP)
(Chunk America/NNP)
(Chunk Applause/NNP)
#chinking with NLTK

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in  tokenized[:20]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            # combine the part of speech tag with a reg exp
            chunkGramm = r"""Chunk:{<.*>+} 
                                            }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGramm)
            chunked = chunkParser.parse(tagged)
            #print the nltk tree
            print(chunked)
            for subtree in chunked.subtrees(filter =lambda t: t.label()== 'Chunk'):
                print(subtree)
            #draw the chunks with  nltk
            chunked.draw()
    except Exception as e:
        print(str(e))
process_content()
(S
  (Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP 'S/POS ADDRESS/NNP)
  BEFORE/IN
  (Chunk A/NNP JOINT/NNP SESSION/NNP)
  OF/IN
  (Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)
  OF/IN
  (Chunk
    THE/NNP
    UNION/NNP
    January/NNP
    31/CD
    ,/,
    2006/CD
    THE/NNP
    PRESIDENT/NNP
    :/:
    Thank/NNP
    you/PRP)
  all/DT
  (Chunk ./.))
(Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP 'S/POS ADDRESS/NNP)
(Chunk A/NNP JOINT/NNP SESSION/NNP)
(Chunk THE/NNP CONGRESS/NNP ON/NNP THE/NNP STATE/NNP)
(Chunk
  THE/NNP
  UNION/NNP
  January/NNP
  31/CD
  ,/,
  2006/CD
  THE/NNP
  PRESIDENT/NNP
  :/:
  Thank/NNP
  you/PRP)
(Chunk ./.)
(S
  (Chunk
    Mr./NNP
    Speaker/NNP
    ,/,
    Vice/NNP
    President/NNP
    Cheney/NNP
    ,/,
    members/NNS)
  of/IN
  (Chunk Congress/NNP ,/, members/NNS)
  of/IN
  the/DT
  (Chunk
    Supreme/NNP
    Court/NNP
    and/CC
    diplomatic/JJ
    corps/NN
    ,/,
    distinguished/JJ
    guests/NNS
    ,/,
    and/CC
    fellow/JJ
    citizens/NNS
    :/:)
  Today/VB
  (Chunk our/PRP$ nation/NN)
  lost/VBD
  a/DT
  beloved/VBN
  (Chunk ,/, graceful/JJ ,/, courageous/JJ woman/NN who/WP)
  called/VBD
  (Chunk America/NNP)
  to/TO
  (Chunk its/PRP$ founding/NN ideals/NNS and/CC)
  carried/VBD
  on/IN
  a/DT
  (Chunk noble/JJ dream/NN ./.))
(Chunk
  Mr./NNP
  Speaker/NNP
  ,/,
  Vice/NNP
  President/NNP
  Cheney/NNP
  ,/,
  members/NNS)
(Chunk Congress/NNP ,/, members/NNS)
(Chunk
  Supreme/NNP
  Court/NNP
  and/CC
  diplomatic/JJ
  corps/NN
  ,/,
  distinguished/JJ
  guests/NNS
  ,/,
  and/CC
  fellow/JJ
  citizens/NNS
  :/:)
(Chunk our/PRP$ nation/NN)
(Chunk ,/, graceful/JJ ,/, courageous/JJ woman/NN who/WP)
(Chunk America/NNP)
(Chunk its/PRP$ founding/NN ideals/NNS and/CC)
(Chunk noble/JJ dream/NN ./.)
(S
  (Chunk Tonight/NN we/PRP)
  are/VBP
  comforted/VBN
  by/IN
  the/DT
  (Chunk hope/NN)
  of/IN
  a/DT
  (Chunk glad/JJ reunion/NN)
  with/IN
  the/DT
  (Chunk husband/NN who/WP)
  was/VBD
  taken/VBN
  (Chunk so/RB long/RB ago/RB ,/, and/CC we/PRP)
  are/VBP
  (Chunk grateful/JJ)
  for/IN
  the/DT
  (Chunk good/JJ life/NN)
  of/IN
  (Chunk Coretta/NNP Scott/NNP King/NNP ./.))
(Chunk Tonight/NN we/PRP)
(Chunk hope/NN)
(Chunk glad/JJ reunion/NN)
(Chunk husband/NN who/WP)
(Chunk so/RB long/RB ago/RB ,/, and/CC we/PRP)
(Chunk grateful/JJ)
(Chunk good/JJ life/NN)
(Chunk Coretta/NNP Scott/NNP King/NNP ./.)
(S (Chunk (/( Applause/NNP ./. )/)))
(Chunk (/( Applause/NNP ./. )/))
(S
  (Chunk President/NNP George/NNP W./NNP Bush/NNP)
  reacts/VBZ
  to/TO
  applause/VB
  during/IN
  (Chunk his/PRP$ State/NNP)
  of/IN
  the/DT
  (Chunk Union/NNP Address/NNP)
  at/IN
  the/DT
  (Chunk Capitol/NNP ,/, Tuesday/NNP ,/, Jan/NNP ./.))
(Chunk President/NNP George/NNP W./NNP Bush/NNP)
(Chunk his/PRP$ State/NNP)
(Chunk Union/NNP Address/NNP)
(Chunk Capitol/NNP ,/, Tuesday/NNP ,/, Jan/NNP ./.)
(S (Chunk 31/CD ,/, 2006/CD ./.))
(Chunk 31/CD ,/, 2006/CD ./.)
(S
  (Chunk White/NNP House/NNP photo/NN)
  by/IN
  (Chunk Eric/NNP DraperEvery/NNP time/NN I/PRP)
  'm/VBP
  (Chunk invited/JJ)
  to/TO
  this/DT
  (Chunk rostrum/NN ,/, I/PRP)
  'm/VBP
  humbled/VBN
  by/IN
  the/DT
  (Chunk privilege/NN ,/, and/CC mindful/NN)
  of/IN
  the/DT
  (Chunk history/NN we/PRP)
  've/VBP
  seen/VBN
  (Chunk together/RB ./.))
(Chunk White/NNP House/NNP photo/NN)
(Chunk Eric/NNP DraperEvery/NNP time/NN I/PRP)
(Chunk invited/JJ)
(Chunk rostrum/NN ,/, I/PRP)
(Chunk privilege/NN ,/, and/CC mindful/NN)
(Chunk history/NN we/PRP)
(Chunk together/RB ./.)
(S
  (Chunk We/PRP)
  have/VBP
  gathered/VBN
  under/IN
  this/DT
  (Chunk Capitol/NNP dome/NN)
  in/IN
  (Chunk moments/NNS)
  of/IN
  (Chunk
    national/JJ
    mourning/NN
    and/CC
    national/JJ
    achievement/NN
    ./.))
(Chunk We/PRP)
(Chunk Capitol/NNP dome/NN)
(Chunk moments/NNS)
(Chunk national/JJ mourning/NN and/CC national/JJ achievement/NN ./.)
(S
  (Chunk We/PRP)
  have/VBP
  served/VBN
  (Chunk America/NNP)
  through/IN
  (Chunk one/CD)
  of/IN
  the/DT
  (Chunk most/RBS consequential/JJ periods/NNS)
  of/IN
  (Chunk our/PRP$ history/NN --/: and/CC it/PRP)
  has/VBZ
  been/VBN
  (Chunk my/PRP$ honor/NN)
  to/TO
  serve/VB
  with/IN
  (Chunk you/PRP ./.))
(Chunk We/PRP)
(Chunk America/NNP)
(Chunk one/CD)
(Chunk most/RBS consequential/JJ periods/NNS)
(Chunk our/PRP$ history/NN --/: and/CC it/PRP)
(Chunk my/PRP$ honor/NN)
(Chunk you/PRP ./.)
(S
  In/IN
  a/DT
  (Chunk system/NN)
  of/IN
  (Chunk
    two/CD
    parties/NNS
    ,/,
    two/CD
    chambers/NNS
    ,/,
    and/CC
    two/CD
    elected/JJ
    branches/NNS
    ,/,
    there/EX
    will/MD
    always/RB)
  be/VB
  (Chunk differences/NNS and/CC debate/NN ./.))
(Chunk system/NN)
(Chunk
  two/CD
  parties/NNS
  ,/,
  two/CD
  chambers/NNS
  ,/,
  and/CC
  two/CD
  elected/JJ
  branches/NNS
  ,/,
  there/EX
  will/MD
  always/RB)
(Chunk differences/NNS and/CC debate/NN ./.)
(S
  (Chunk But/CC even/RB tough/JJ debates/NNS can/MD)
  be/VB
  conducted/VBN
  in/IN
  a/DT
  (Chunk
    civil/JJ
    tone/NN
    ,/,
    and/CC
    our/PRP$
    differences/NNS
    can/MD
    not/RB)
  be/VB
  allowed/VBN
  to/TO
  harden/VB
  into/IN
  (Chunk anger/NN ./.))
(Chunk But/CC even/RB tough/JJ debates/NNS can/MD)
(Chunk
  civil/JJ
  tone/NN
  ,/,
  and/CC
  our/PRP$
  differences/NNS
  can/MD
  not/RB)
(Chunk anger/NN ./.)
(S
  To/TO
  confront/VB
  the/DT
  (Chunk great/JJ issues/NNS)
  before/IN
  (Chunk us/PRP ,/, we/PRP must/MD)
  act/VB
  in/IN
  a/DT
  (Chunk spirit/NN)
  of/IN
  (Chunk goodwill/NN and/CC respect/NN)
  for/IN
  (Chunk one/CD)
  another/DT
  (Chunk --/: and/CC I/PRP will/MD)
  do/VB
  (Chunk my/PRP$ part/NN ./.))
(Chunk great/JJ issues/NNS)
(Chunk us/PRP ,/, we/PRP must/MD)
(Chunk spirit/NN)
(Chunk goodwill/NN and/CC respect/NN)
(Chunk one/CD)
(Chunk --/: and/CC I/PRP will/MD)
(Chunk my/PRP$ part/NN ./.)
(S
  (Chunk Tonight/NNP)
  the/DT
  (Chunk state/NN)
  of/IN
  (Chunk our/PRP$ Union/NNP)
  is/VBZ
  (Chunk strong/JJ --/: and/CC together/RB we/PRP will/MD)
  make/VB
  (Chunk it/PRP stronger/JJR ./.))
(Chunk Tonight/NNP)
(Chunk state/NN)
(Chunk our/PRP$ Union/NNP)
(Chunk strong/JJ --/: and/CC together/RB we/PRP will/MD)
(Chunk it/PRP stronger/JJR ./.)
(S (Chunk (/( Applause/NNP ./. )/)))
(Chunk (/( Applause/NNP ./. )/))
(S
  In/IN
  this/DT
  (Chunk decisive/JJ year/NN ,/, you/PRP and/CC I/PRP will/MD)
  make/VB
  (Chunk choices/NNS that/WDT)
  determine/VBP
  both/DT
  the/DT
  (Chunk future/NN and/CC)
  the/DT
  (Chunk character/NN)
  of/IN
  (Chunk our/PRP$ country/NN ./.))
(Chunk decisive/JJ year/NN ,/, you/PRP and/CC I/PRP will/MD)
(Chunk choices/NNS that/WDT)
(Chunk future/NN and/CC)
(Chunk character/NN)
(Chunk our/PRP$ country/NN ./.)
(S
  (Chunk We/PRP will/MD)
  choose/VB
  to/TO
  act/VB
  (Chunk confidently/RB)
  in/IN
  pursuing/VBG
  the/DT
  (Chunk enemies/NNS)
  of/IN
  (Chunk freedom/NN --/: or/CC retreat/NN)
  from/IN
  (Chunk our/PRP$ duties/NNS)
  in/IN
  the/DT
  (Chunk hope/NN)
  of/IN
  an/DT
  (Chunk easier/JJR life/NN ./.))
(Chunk We/PRP will/MD)
(Chunk confidently/RB)
(Chunk enemies/NNS)
(Chunk freedom/NN --/: or/CC retreat/NN)
(Chunk our/PRP$ duties/NNS)
(Chunk hope/NN)
(Chunk easier/JJR life/NN ./.)
(S
  (Chunk We/PRP will/MD)
  choose/VB
  to/TO
  build/VB
  (Chunk our/PRP$ prosperity/NN)
  by/IN
  leading/VBG
  the/DT
  (Chunk world/NN economy/NN --/: or/CC)
  shut/VB
  (Chunk ourselves/PRP off/RP)
  from/IN
  (Chunk trade/NN and/CC opportunity/NN ./.))
(Chunk We/PRP will/MD)
(Chunk our/PRP$ prosperity/NN)
(Chunk world/NN economy/NN --/: or/CC)
(Chunk ourselves/PRP off/RP)
(Chunk trade/NN and/CC opportunity/NN ./.)
(S
  In/IN
  a/DT
  (Chunk complex/JJ and/CC challenging/JJ time/NN ,/,)
  the/DT
  (Chunk road/NN)
  of/IN
  (Chunk isolationism/NN and/CC protectionism/NN may/MD)
  seem/VB
  (Chunk broad/JJ and/CC inviting/NN --/: yet/CC it/PRP)
  ends/VBZ
  in/IN
  (Chunk danger/NN and/CC decline/NN ./.))
(Chunk complex/JJ and/CC challenging/JJ time/NN ,/,)
(Chunk road/NN)
(Chunk isolationism/NN and/CC protectionism/NN may/MD)
(Chunk broad/JJ and/CC inviting/NN --/: yet/CC it/PRP)
(Chunk danger/NN and/CC decline/NN ./.)
(S
  The/DT
  (Chunk only/JJ way/NN)
  to/TO
  protect/VB
  (Chunk our/PRP$ people/NNS ,/,)
  the/DT
  (Chunk only/JJ way/NN)
  to/TO
  secure/VB
  the/DT
  (Chunk peace/NN ,/,)
  the/DT
  (Chunk only/JJ way/NN)
  to/TO
  control/VB
  (Chunk our/PRP$ destiny/NN)
  is/VBZ
  by/IN
  (Chunk our/PRP$ leadership/NN --/:)
  so/IN
  the/DT
  (Chunk United/NNP States/NNPS)
  of/IN
  (Chunk America/NNP will/MD)
  continue/VB
  to/TO
  lead/VB
  (Chunk ./.))
(Chunk only/JJ way/NN)
(Chunk our/PRP$ people/NNS ,/,)
(Chunk only/JJ way/NN)
(Chunk peace/NN ,/,)
(Chunk only/JJ way/NN)
(Chunk our/PRP$ destiny/NN)
(Chunk our/PRP$ leadership/NN --/:)
(Chunk United/NNP States/NNPS)
(Chunk America/NNP will/MD)
(Chunk ./.)
(S (Chunk (/( Applause/NNP ./. )/)))
(Chunk (/( Applause/NNP ./. )/))
def process_content():
    try:
        for i in  tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged,binary =True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))
process_content()
global name 'tokenized' is not defined
import random
import nltk
nltk.download()
from nltk.corpus import movie_reviews
showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
#build list of docs
documents = [(list(movie_reviews.words(fileid)),category)
            for category  in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

#shuffle docs
random.shuffle(documents)
print("number of docs:{}".format(len(documents)))
print("first review:{}".format(documents[0]))
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
print("most common words:{}".format(all_words.most_common(15)))
print("the word happy:{}".format(all_words["happy"]))
number of docs:2000
first review:([u'it', u"'", u's', u'a', u'curious', u'thing', u'-', u'i', u"'", u've', u'found', u'that', u'when', u'willis', u'is', u'not', u'called', u'on', u'to', u'carry', u'the', u'whole', u'movie', u',', u'he', u"'", u's', u'much', u'better', u'and', u'so', u'is', u'the', u'movie', u'.', u'even', u'though', u',', u'in', u'the', u'sixth', u'sense', u'he', u'is', u'the', u'"', u'name', u'"', u',', u'he', u'doesn', u"'", u't', u'have', u'the', u'pivotal', u'role', u'.', u'that', u'honour', u'goes', u'to', u'haley', u'osment', u'who', u'plays', u'cole', u'sear', u'(', u'cute', u'pun', u',', u'seer', u')', u'a', u'9', u'year', u'old', u'boy', u'who', u'can', u'see', u'ghosts', u'.', u'if', u'osment', u'was', u'cute', u'or', u'precious', u',', u'the', u'director', u'going', u'for', u'the', u'maudlin', u',', u'this', u'would', u'be', u'nothing', u'more', u'than', u'a', u'movie', u'-', u'of', u'-', u'the', u'-', u'week', u',', u'thankfully', u',', u'osment', u'is', u'not', u'only', u'better', u'than', u'that', u',', u'but', u'in', u'some', u'instances', u',', u'blows', u'everyone', u'else', u'off', u'the', u'screen', u'in', u'a', u'bravura', u'performance', u'.', u'we', u'get', u'to', u'see', u'his', u'fears', u',', u'vulnerabilities', u',', u'strengths', u'and', u'intelligence', u'which', u'makes', u'the', u'sixth', u'sense', u'one', u'of', u'the', u'best', u'movies', u'i', u"'", u've', u'seen', u'this', u'year', u'.', u'the', u'whole', u'cast', u'matches', u'him', u'in', u'quality', u',', u'with', u'willis', u'giving', u'a', u'fairly', u'low', u'key', u'performance', u'that', u'matches', u'the', u'subject', u'matter', u'.', u'one', u'thing', u'about', u'this', u'movie', u',', u'its', u'target', u'.', u'this', u'isn', u"'", u't', u'a', u'sfxfest', u'like', u'the', u'haunting', u'or', u'a', u'gorefest', u',', u'this', u'is', u'more', u'what', u'i', u"'", u'd', u'call', u'a', u'supernatural', u'drama', u',', u'more', u'interested', u'in', u'characters', u'than', u'in', u'dazzling', u'you', u'with', u'makeup', u'.', u'one', u'caveat', u':', u'there', u"'", u's', u'a', u'lovely', u'twist', u'in', u'the', u'movie', u',', u'something', u'like', u'the', u'usual', u'suspects', u',', u'where', u'you', u'end', u'up', u'replaying', u'the', u'movie', u'in', u'your', u'head', u'rethinking', u'what', u'you', u'have', u'just', u'seen', u'.', u'i', u'was', u'extremely', u'lucky', u'to', u'see', u'it', u'as', u'a', u'sneak', u'preview', u'in', u'toronto', u',', u'before', u'any', u'hype', u'or', u'critical', u'reviews', u'were', u'out', u',', u'so', u'i', u'went', u'in', u'with', u'no', u'biases', u'.', u'if', u'anyone', u'want', u'to', u'talk', u'to', u'about', u'the', u'movie', u'before', u'you', u'see', u'it', u',', u'don', u"'", u't', u'let', u'them', u'.', u'let', u'the', u'director', u'explain', u'on', u'his', u'own', u'pace', u'and', u'you', u"'", u'll', u'enjoy', u'the', u'movie', u'vastly', u'more', u'.'], u'pos')
most common words:[(u',', 77717), (u'the', 76529), (u'.', 65876), (u'a', 38106), (u'and', 35576), (u'of', 34123), (u'to', 31937), (u"'", 30585), (u'is', 25195), (u'in', 21822), (u's', 18513), (u'"', 17612), (u'it', 16107), (u'that', 15924), (u'-', 15595)]
the word happy:215
print(len(all_words))
39768
# well use teh 4000 most common words as features
word_features = list(all_words.keys())[:4000]
#build a find_features function that will determine which of the 4000 word features are contained in a review
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#lets  use an example from a neg review:
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))
for key,value in features.items():
    if value == True:
        print key
    
     
shows
kids
music
want
production
feeling
away
.
has
confusing
bottom
exact
years
still
now
didn
one
s
world
arrow
with
concept
7
horror
more
visions
american
feels
also
into
video
makes
start
tons
despite
meantime
'
insight
off
not
wes
problem
print(features)
{u'sitters': False, u'clamoring': False, u'madsen': False, u'sonja': False, u'unsworth': False, u'woods': False, u'spiders': False, u'gavan': False, u'francesco': False, u'francesca': False, u'fedoore': False, u'comically': False, u'negg': False, u'localized': False, u'guelph': False, u'stinks': False, u'disobeying': False, u'hennings': False, u'porno': False, u'canet': False, u'ceases': False, u'giacomo': False, u'stinky': False, u'scold': False, u'originality': False, u'neighbours': False, u'caned': False, u'intros': False, u'rickman': False, u'worth': False, u'porns': False, u'alternating': False, u'amorous': False, u'copasetic': False, u'slothful': False, u'wracked': False, u'dougnac': False, u'aurora': False, u'stipulate': False, u'kissed_': False, u'helgenberger': False, u'soldering': False, u'capoeira': False, u'rosalba': False, u'crackin': False, u'rawhide': False, u'summarized': False, u'waterlogged': False, u'screaming': False, u'bushido': False, u'yikes': False, u'recollections': False, u'liaisons': False, u'grueling': False, u'sommerset': False, u'investigator': False, u'wooden': False, u'wednesday': False, u'broiled': False, u'samurai': False, u'circuitry': False, u'notifying': False, u'crotch': False, u'elgar': False, u'errol': False, u'stereotypical': False, u'monologue': False, u'shows': True, u'roldan': False, u'jamaica': False, u'bazooms': False, u'betsy': False, u'sabbato': False, u'snuggles': False, u'hanging': False, u'pescara': False, u'feasibility': False, u'miniatures': False, u'nerdiest': False, u'advantaged': False, u'mesmerising': False, u'gorman': False, u'woody': False, u'consenting': False, u'scraped': False, u'gazon': False, u'machines': False, u'inanimate': False, u'errors': False, u'euclidean': False, u'rekindle': False, u'offshoots': False, u'cooking': False, u'fonzie': False, u'opportunists': False, u'petri': False, u'videodrome': False, u'outfielders': False, u'numeral': False, u'succumb': False, u'shocks': False, u'personifies': False, u'viewings': False, u'chins': False, u'crooned': False, u'jubilantly': False, u'rocque': False, u'spunky': False, u'dilapidating': False, u'equals': False, u'metaphorically': False, u'boyum': False, u'ching': False, u'protection': False, u'china': False, u'personified': False, u'dobie': False, u'shandling': False, u'wiseguy': False, u'natured': False, u'watermelons': False, u'kids': True, u'uplifting': False, u'storywise': False, u'k': False, u'controversy': False, u'rebhorn': False, u'crowdpleasing': False, u'stressed': False, u'neurologist': False, u'bunker': False, u'spotty': False, u'climber': False, u'appropriately': False, u'cobblers': False, u'projection': False, u'urbaniak': False, u'outraging': False, u'brs': False, u'lengthen': False, u'emerich': False, u'flotsam': False, u'bro': False, u'lavatory': False, u'archaeological': False, u'unsinkable': False, u'stern': False, u'compulsively': False, u'namuth': False, u'kethcum': False, u'sarah': False, u'devastation': False, u'plow': False, u'dna': False, u'plop': False, u'catchy': False, u'insecurity': False, u'sweater': False, u'coins': False, u'ploy': False, u'cannibal': False, u'sidebars': False, u'_people_': False, u'music': True, u'therefore': False, u'superweapons': False, u'mutinies': False, u'administering': False, u'guinevere': False, u'magyuver': False, u'separated': False, u'deloreans': False, u'bombast': False, u'paperwork': False, u'kohn': False, u'mesmerize': False, u'ascribe': False, u'yahoo': False, u'exuberantly': False, u'======': False, u'championships': False, u'boorman': False, u'diggler': False, u'provide': False, u'foregrounds': False, u'primeval': False, u'conditioned': False, u'voicework': False, u'blocking': False, u'circumstances': False, u'reingold': False, u'tastefully': False, u'1993': False, u'morally': False, u'locked': False, u'1994': False, u'daqughter': False, u'1996': False, u'1999': False, u'1998': False, u'overpowers': False, u'cuddly': False, u'divorces': False, u'locker': False, u'tissue': False, u'locket': False, u'era': False, u'soundbite': False, u'gershon': False, u'elbow': False, u'erm': False, u'ern': False, u'plunges': False, u'recipes': False, u'scripting': False, u'matilda': False, u'phrase': False, u'transition': False, u'wang': False, u'indicated': False, u'wane': False, u'portorican': False, u'flung': False, u'winnfield': False, u'heartless': False, u'strangelove': False, u'titanium': False, u'dishearteningly': False, u'want': True, u'repressiveness': False, u'pinto': False, u'absolute': False, u'impassive': False, u'augustin': False, u'skyler': False, u'vicent': False, u'beyer': False, u'travel': False, u'nuts': False, u'copious': False, u'kyzynski': False, u'recovers': False, u'playback': False, u'dangerfield': False, u'conn': False, u'moron': False, u'titillate': False, u'prostitues': False, u'truthful': False, u'cadence': False, u'thivisol': False, u'sonorra': False, u'sordidness': False, u'henreid': False, u'invitation': False, u'memorial': False, u'customs': False, u'millimeter': False, u'dinosaurs': False, u'wrong': False, u'walton': False, u'aurore': False, u'ladden': False, u'cerebrally': False, u'sentencing': False, u'dumbo': False, u'greediness': False, u'arch': False, u'foundering': False, u'hurricaine': False, u'prying': False, u'complacent': False, u'colorfully': False, u'elusive': False, u'glenne': False, u'alienate': False, u'recombination': False, u'schneider': False, u'appreciate': False, u'americanization': False, u'subplots': False, u'purging': False, u'sickening': False, u'tulip': False, u'18th': False, u'davies': False, u'moroder': False, u'nonsensical': False, u'romper': False, u'disengaging': False, u'droagon': False, u'_american_psycho_': False, u'leeanne': False, u'snugly': False, u'gumption': False, u'kuei': False, u'scalding': False, u'welcomed': False, u'matewan': False, u'concurrence': False, u'homeboys': False, u'stoicism': False, u'whizzing': False, u'wachowskis': False, u'matsumoto': False, u'sidekicks': False, u'vito': False, u'innovative': False, u'rewarded': False, u'welcomes': False, u'understates': False, u'wickedly': False, u'fit': False, u'lifeline': False, u'bringing': False, u'fix': False, u'inspections': False, u'albums': False, u'christie': False, u'_i_know_what_you_did_last_summer_': False, u'trager': False, u'production': True, u'understated': False, u'fig': False, u'valor': False, u'wales': False, u'fin': False, u'pantaeva': False, u'travellers': False, u'vamires': False, u'soundbites': False, u'songwriter': False, u'municipality': False, u'locusts': False, u'safe': False, u'inhereit': False, u'collide': False, u'laconically': False, u'expressionist': False, u'commemorate': False, u'roommate': False, u'guiler': False, u'effects': False, u'expressionism': False, u'cohagen': False, u'multidimensional': False, u'sixteen': False, u'undeveloped': False, u'saddened': False, u'aneeka': False, u'petrovsky': False, u'defeaningly': False, u'progression': False, u'dingo': False, u'whacking': False, u'bartok': False, u'reasonably': False, u'routines': False, u'barton': False, u'l': False, u'foregin': False, u'dingy': False, u'sugarplums': False, u'nighthawks': False, u'frewer': False, u'tenets': False, u'averse': False, u'ingrid': False, u'fearsomely': False, u'feeds': False, u'_eve': False, u'overpopulated': False, u'telescope': False, u'justice': False, u'dumping': False, u'masseuse': False, u'allah': False, u'allan': False, u'sled': False, u'parasites': False, u'maniacs': False, u'tragicomic': False, u'slew': False, u'roadblock': False, u'inadvertently': False, u'touts': False, u'oprah': False, u'smirk': False, u'scrumptiously': False, u'indiscretion': False, u'maduro': False, u'nordoff': False, u'danforth': False, u'mason': False, u'encourage': False, u'daniel': False, u'adapt': False, u'uuuhhmmm': False, u'confections': False, u'zellwegger': False, u'judith': False, u'outburst': False, u'mullen': False, u'abbott': False, u'stamping': False, u'meatier': False, u'abbots': False, u'barrier': False, u'colorless': False, u'leftovers': False, u'beristain': False, u'bitchin': False, u'pumpkins': False, u'corrects': False, u'forcibly': False, u'crowned': False, u'estimate': False, u'universally': False, u'chlorine': False, u'renee': False, u'chortled': False, u'dammit': False, u'birdie': False, u'r2': False, u'sickeningly': False, u'refugee': False, u'locomotive': False, u'flawless': False, u'renew': False, u'disturbed': False, u'competed': False, u'dentures': False, u'loudness': False, u'footsteps': False, u'juergen': False, u'haunted': False, u'render': False, u'elmo': False, u'disfigured': False, u'railroads': False, u'immolation': False, u'procreate': False, u'dog_': False, u'stylistics': False, u'kfc': False, u'megabytes': False, u'antarctic': False, u'electronic': False, u'mojo': False, u'sooty': False, u'timothy': False, u'olds': False, u'renovated': False, u'service': False, u'forrester': False, u'2058': False, u'corsucant': False, u'ingen': False, u'reuben': False, u'approximately': False, u'needed': False, u'blurts': False, u'master': False, u'_2001_': False, u'cassavetes': False, u'mistic': False, u'critter': False, u'john': False, u'genesis': False, u'weendigo': False, u'caitlyn': False, u'rewards': False, u'paltry': False, u'enthrall': False, u'offhand': False, u'oingo': False, u'lunges': False, u'waster': False, u'wastes': False, u'scorpions': False, u'mutilated': False, u'enmeshed': False, u'cereal': False, u'wasted': False, u'downtime': False, u'anachronisms': False, u'positively': False, u'nbk': False, u'ahmed': False, u'reclining': False, u'guile': False, u'bannister': False, u'roelfs': False, u'handcuffs': False, u'idly': False, u'project': False, u'idle': False, u'exclaimed': False, u'guilt': False, u'friend': False, u'historical': False, u'apparantly': False, u'rienfenstal': False, u'feeling': True, u'seminal': False, u'humble': False, u'unrest': False, u'rouen': False, u'tautness': False, u'longs': False, u'portraits': False, u'sustaining': False, u'flavorful': False, u'spectrum': False, u'palma': False, u'enchanted': False, u'longo': False, u'refreshingly': False, u'arousal': False, u'tenuous': False, u'urinate': False, u'contents': False, u'dozen': False, u'affairs': False, u'sitations': False, u'wholesome': False, u'courier': False, u'scumbagginess': False, u'cronenberg': False, u'convenient': False, u'uncouth': False, u'gripe': False, u'saunder': False, u'hamper': False, u'racers': False, u'pittsburgh': False, u'toothed': False, u'subjects': False, u'nuez': False, u'thundering': False, u'pilgrimage': False, u'workmates': False, u'enervation': False, u'shipments': False, u'gravy': False, u'germann': False, u'committing': False, u'caprenter': False, u'bruce': False, u'limitless': False, u'diminishing': False, u'vexing': False, u'cinematic': False, u'resonates': False, u'ramblings': False, u'disjointed': False, u'stardom': False, u'mouth': False, u'culminates': False, u'reverence': False, u'scripts': False, u'resonated': False, u'utilize': False, u'competing': False, u'parting': False, u'expound': False, u'singer': False, u'macfadyen': False, u'bracken': False, u'exhilaratingly': False, u'dubarry': False, u'musical': False, u'multiracial': False, u'swamp': False, u'bracket': False, u'tech': False, u'fugitives': False, u'keeble': False, u'rayden': False, u'rhythmless': False, u'brinkford': False, u'germany': False, u'scream': False, u'crowbar': False, u'saying': False, u'rockies': False, u'overdoes': False, u'stephens': False, u'lewis': False, u'teresa': False, u'loitered': False, u'rosselini': False, u'padded': False, u'bellow': False, u'disturbs': False, u'ulcer': False, u'alferd': False, u'tempted': False, u'cheaply': False, u'councilmembers': False, u'junkyard': False, u'hounded': False, u'capitol': False, u'orleans': False, u'geyser': False, u'clicked': False, u'hedley': False, u'friendliest': False, u'quaint': False, u'nullifies': False, u'grosbard': False, u'rico': False, u'cued': False, u'bliss': False, u'rick': False, u'rich': False, u'rice': False, u'nullified': False, u'rectangle': False, u'rica': False, u'monsieur': False, u'plate': False, u'cappuccino': False, u'joely': False, u'tramell': False, u'belaboured': False, u'chaney': False, u'uncountable': False, u'designing': False, u'plath': False, u'psychiatric': False, u'wisecracks': False, u'platt': False, u'imitating': False, u'clumsiness': False, u'roundabout': False, u'altogether': False, u'chyron': False, u'vividly': False, u'beleive': False, u'droning': False, u'vicariously': False, u'runs': False, u'stoically': False, u'plotholia': False, u'spilling': False, u'nicely': False, u'boarder': False, u'pretzel': False, u'patch': False, u'eyelids': False, u'rahad': False, u'rune': False, u'gears': False, u'rung': False, u'krupa': False, u'boarded': False, u'scrubbed': False, u'secretary': False, u'jovivich': False, u'heirloom': False, u'clarified': False, u'claymation': False, u'sensitivity': False, u'pinon': False, u'slashfest': False, u'horrendous': False, u'discussions': False, u'humpalot': False, u'optimum': False, u'hairbrush': False, u'techniques': False, u'develop': False, u'pastel': False, u'48th': False, u'playfulness': False, u'pressured': False, u'deadpan': False, u'pasted': False, u'away': True, u'irs': False, u'droves': False, u'bandaras': False, u'collette': False, u'bracing': False, u'arcane': False, u'arcand': False, u'meditative': False, u'ira': False, u'drawl': False, u'encounters': False, u'ire': False, u'huison': False, u'extend': False, u'nature': False, u'handful': False, u'lapping': False, u'transylvanians': False, u'diesl': False, u'taraji': False, u'gays': False, u'succumbs': False, u'extent': False, u'beelzebub': False, u'reacquaints': False, u'kitchen': False, u'tyranny': False, u'climate': False, u'benigness': False, u'psychologists': False, u'dorff': False, u'veer': False, u'disdain': False, u'voyeuristic': False, u'himalayas': False, u'compton': False, u'askew': False, u'orwell': False, u'hollywoodization': False, u'lookin': False, u'disappears': False, u'fearlessly': False, u'eradicate': False, u'zigged': False, u'rehash': False, u'mortified': False, u'tone': False, u'maclaine': False, u'layers': False, u'murtaugh': False, u'upbeats': False, u'sobchak': False, u'excess': False, u'gopher': False, u'gypsies': False, u'fondled': False, u'charnel': False, u'lick': False, u'affiliated': False, u'tony': False, u'surname': False, u'blonde': False, u'diabolically': False, u'telecommunications': False, u'priscilla': False, u'underdone': False, u'downtrodden': False, u'milquetoast': False, u'wright': False, u'union': False, u'fro': False, u'resemblances': False, u'.': True, u'muck': False, u'polemic': False, u'much': False, u'wyman': False, u'noel': False, u'tonino': False, u'kanoby': False, u'fry': False, u'toning': False, u'ocious': False, u'obese': False, u'chinlund': False, u'superpowers': False, u'retrospect': False, u'spit': False, u'attacked': False, u'arkin': False, u'excite': False, u'freehold': False, u'almasy': False, u'psychically': False, u'comprehensible': False, u'dave': False, u'yugoslavians': False, u'doubts': False, u'clairvoyant': False, u'spin': False, u'takaaki': False, u'diverted': False, u'righteous': False, u'espoused': False, u'professionally': False, u'paraglider': False, u'employ': False, u'nfeatured': False, u'misconstrued': False, u'loki': False, u'thrash': False, u'prostrate': False, u'35th': False, u'characterizing': False, u'blackly': False, u'cont': False, u'krays': False, u'canoeing': False, u'beetles': False, u'ditching': False, u'communicating': False, u'verges': False, u'lackies': False, u'separatist': False, u'tylenol': False, u'mirabella': False, u'eighteen': False, u'cong': False, u'haplessly': False, u'voges': False, u'oxymoron': False, u'turner': False, u'sever': False, u'hone': False, u'protovision': False, u'hong': False, u'inventively': False, u'portobello': False, u'remand': False, u'mummified': False, u'amount': False, u'honk': False, u'windsor': False, u'writerly': False, u'spews': False, u'alevey': False, u'split': False, u'synch': False, u'mindfuck': False, u'codename': False, u'principals': False, u'cavanaugh': False, u'advertising': False, u'wheel': False, u'boiled': False, u'effortlessly': False, u'fuss': False, u'issac': False, u'frenchmen': False, u'hana': False, u'vivien': False, u'torpedoes': False, u'lyndon': False, u'bening': False, u'liberties': False, u'marched': False, u'buliwyf': False, u'boiler': False, u'dashing': False, u'rulebook': False, u'hans': False, u'selfless': False, u'brainers': False, u'featherweight': False, u'postponed': False, u'whereby': False, u'noblewoman': False, u'1600': False, u'fashionable': False, u'mentors': False, u'academic': False, u'stillness': False, u'academia': False, u'goofing': False, u'humbly': False, u'sullenly': False, u'waitering': False, u'corporate': False, u'massaging': False, u'pronouncements': False, u'gigolo': False, u'solaris': False, u'belloq': False, u'absurdities': False, u'golden': False, u'vying': False, u'newton': False, u'_would_': False, u'homogeneity': False, u'snickered': False, u'sabbatical': False, u'ol': False, u'portrayed': False, u'electronically': False, u'lasso': False, u'hai': False, u'denton': False, u'designers': False, u'hal': False, u'ham': False, u'han': False, u'cornell': False, u'similarities': False, u'hab': False, u'espouses': False, u'had': False, u'insubordination': False, u'hag': False, u'jost': False, u'hay': False, u'mcnamara': False, u'cognac': False, u'unwanted': False, u'beloved': False, u'joss': False, u'hap': False, u'dion': False, u'har': False, u'has': True, u'hat': False, u'preciously': False, u'hav': False, u'haw': False, u'yoko': False, u'packin': False, u'insensitive': False, u'elders': False, u'survival': False, u'tricking': False, u'inflicting': False, u'unequivocally': False, u'youngstein': False, u'otherworldly': False, u'indicative': False, u'everton': False, u'shadow': False, u'vapors': False, u'unfounded': False, u'ballhaus': False, u'hairless': False, u'sleuthing': False, u'eroded': False, u'arcs': False, u'deviance': False, u'cooler': False, u'huns': False, u'alice': False, u'noteables': False, u'festivities': False, u'sorvino': False, u'homing': False, u'night': False, u'revisiting': False, u'grotesquely': False, u'cooled': False, u'misdemeanors': False, u'unabashedly': False, u'attorney': False, u'dimitri': False, u'crowd': False, u'crowe': False, u'czech': False, u'flatter': False, u'mosques': False, u'crown': False, u'hypsy': False, u'deflection': False, u'changwei': False, u'captive': False, u'couture': False, u'stardust': False, u'flatten': False, u'kieslowski': False, u'billboard': False, u'bore': False, u'confusing': True, u'adorably': False, u'congratulate': False, u'born': False, u'wiseacre': False, u'peerless': False, u'bottom': True, u'chabert': False, u'phillipie': False, u'inhuman': False, u'plucked': False, u'asking': False, u'absolution': False, u'lahore': False, u'melange': False, u'monogamy': False, u'seagrave': False, u'hippest': False, u'participation': False, u'subkoff': False, u'unequipped': False, u'peek': False, u'rooker': False, u'peel': False, u'sadie': False, u'elucidate': False, u'barcode': False, u'shogun': False, u'eduard': False, u'starring': False, u'tribes': False, u'peer': False, u'guild': False, u'peep': False, u'disdains': False, u'explainable': False, u'peet': False, u'menage': False, u'stoker': False, u'deathless': False, u'ferraris': False, u'caraciture': False, u'restlessness': False, u'benches': False, u'_____': False, u'filmcritic': False, u'capades': False, u'bicentennial': False, u'oneness': False, u'mussenden': False, u'janitorial': False, u'hoenicker': False, u'stoked': False, u'whoaaaaaa': False, u'shared': False, u'kilgore': False, u'gads': False, u'dahlings': False, u'jacques': False, u'soc': False, u'guiness': False, u'8034': False, u'wasting': False, u'maxwell': False, u'marshall': False, u'honeymoon': False, u'profession': False, u'mba': False, u'liebes': False, u'rendering': False, u'beings': False, u'raiser': False, u'marshals': False, u'hallucinogenic': False, u'shoots': False, u'aggressivelly': False, u'stumble': False, u'intervention': False, u'familiarize': False, u'despised': False, u'deception': False, u'fabric': False, u'plod': False, u'suffice': False, u'altitude': False, u'unfocused': False, u'raped': False, u'golberg': False, u'grasping': False, u'despises': False, u'obserable': False, u'greatness': False, u'rapes': False, u'exacty': False, u'grooms': False, u'spurting': False, u'overjoyed': False, u'ballisitic': False, u'needles': False, u'catalyst': False, u'congratulations': False, u'humbled': False, u'masquerading': False, u'ancestral': False, u'maximum': False, u'flirtatiously': False, u'smashes': False, u'1600s': False, u'humbler': False, u'maximus': False, u'complications': False, u'exacts': False, u'smashed': False, u'verging': False, u'duet': False, u'@$&%': False, u'dues': False, u'clarke': False, u'passenger': False, u'153': False, u'ayla': False, u'disgrace': False, u'barrymore': False, u'minah': False, u'unnerve': False, u'yankovich': False, u'borg': False, u'guessed': False, u'expertise': False, u'potholes': False, u'decapitation': False, u'empath': False, u'paglia': False, u'triangles': False, u'newlyweds': False, u'slurring': False, u'leer': False, u'spacemusic': False, u'biederman': False, u'paralells': False, u'dowling': False, u'cambodia': False, u'fuse': False, u'pasadena': False, u'role': False, u'telefixated': False, u'evasive': False, u'rolf': False, u'test': False, u'vegetative': False, u'wordlessly': False, u'roll': False, u'intend': False, u'palms': False, u'irma': False, u'orlock': False, u'mulling': False, u'update': False, u'transported': False, u'palme': False, u'connote': False, u'enthused': False, u'comely': False, u'intent': False, u'smelling': False, u'variable': False, u'batmans': False, u'detox': False, u'bacri': False, u'hawkes': False, u'explosions': False, u'loren': False, u'meteorologist': False, u'shootout': False, u'domingo': False, u'highness': False, u'faze': False, u'gheorghe': False, u"']": False, u'interval': False, u'concorde': False, u'beds': False, u'pileggi': False, u'accolade': False, u'banquets': False, u'gutting': False, u'ules': False, u'overturned': False, u'nighshift': False, u'songs': False, u'gown': False, u'hyperdrive': False, u'childs': False, u'cincinnati': False, u'chain': False, u'whoever': False, u'separates': False, u'silverware': False, u'horseback': False, u'supplement': False, u'synapses': False, u'bandits': False, u'unraveled': False, u'battle': False, u'chair': False, u'macht': False, u'ballet': False, u'malintentioned': False, u'grapples': False, u'graph': False, u'freelance': False, u'zeroing': False, u'crates': False, u'crater': False, u'continute': False, u'silencers': False, u'aristocratic': False, u'dissatisfied': False, u'quilty': False, u'obssessed': False, u'macho': False, u'oversight': False, u'extols': False, u'periphery': False, u'tenacious': False, u'1991': False, u'downloading': False, u'paychecks': False, u'icing': False, u'jerk': False, u'tnt': False, u'jere': False, u'prancer': False, u'prances': False, u'choice': False, u'aissa': False, u'embark': False, u'gloomy': False, u'ghostbusters': False, u'stays': False, u'1995': False, u'airing': False, u'exact': True, u'minute': False, u'discovered': False, u'cooks': False, u'scholl': False, u'flammable': False, u'1997': False, u'mcconaughay': False, u'skulduggery': False, u'adorable': False, u'detonated': False, u'masturbates': False, u'gun': False, u'minnie': False, u'skewed': False, u'mathew': False, u'stupendously': False, u'coincidences': False, u'evades': False, u'gut': False, u'skewer': False, u'guy': False, u'matthau': False, u'conglomeration': False, u'definable': False, u'ferocious': False, u'xenophobe': False, u'poledouris': False, u'dryburgh': False, u'trails': False, u'agitate': False, u'pugilistic': False, u'heavyweight': False, u'cost': False, u'chopping': False, u'shirts': False, u'ogled': False, u'biopic': False, u'headset': False, u'lavishness': False, u'recommeded': False, u'massironi': False, u'barbie': False, u'restfulness': False, u'antwerp': False, u'celebrated': False, u'rotunno': False, u'karras': False, u'shares': False, u'peels': False, u'petaluma': False, u'tizard': False, u'goldblum': False, u'celebrates': False, u'cynically': False, u'unintentionally': False, u'paradoxical': False, u'drafted': False, u'erb': False, u'oldies': False, u'climbs': False, u'blunted': False, u'topicality': False, u'sparky': False, u'gladys': False, u'giles': False, u'caesella': False, u'underseen': False, u'address': False, u'dwindling': False, u'teaches': False, u'teacher': False, u'mcfadden': False, u'benson': False, u'mafioso': False, u'accomplishes': False, u'dusty': False, u'impacted': False, u'premieres': False, u'cusack': False, u'accomplished': False, u'sprouted': False, u'franklin': False, u'reagents': False, u'expressively': False, u'enrols': False, u'influx': False, u'plotted': False, u'monty': False, u'kasinsky': False, u'regardless': False, u'architectural': False, u'extra': False, u'houseman': False, u'substitution': False, u'sextet': False, u'uphill': False, u'betraying': False, u'jose': False, u'pees': False, u'fbi': False, u'nauseating': False, u'firearm': False, u'fakery': False, u'hallie': False, u'darnell': False, u'undergone': False, u'working': False, u'cohesiveness': False, u'oldham': False, u'captivate': False, u'quivering': False, u'symbolizes': False, u'unctuous': False, u'coalesce': False, u'wonderfully': False, u'opposed': False, u'unjust': False, u'symbolized': False, u'familar': False, u'perishes': False, u'rainfall': False, u'abrams': False, u'ooooooo': False, u'imrie': False, u'assimilation': False, u'nguyen': False, u'consoles': False, u'sanctimony': False, u'relentlessy': False, u'riders': False, u'barcalow': False, u'warchus': False, u'rebounding': False, u'150th': False, u'ghidrah': False, u'decked': False, u'pooper': False, u'originally': False, u'dmitri': False, u'abortion': False, u'americanised': False, u'reportage': False, u'decker': False, u'harmonious': False, u'remaning': False, u'following': False, u'woefully': False, u'zippers': False, u'trans': False, u'admired': False, u'fracturing': False, u'mirrors': False, u'stetson': False, u'chip': False, u'chit': False, u'parachute': False, u'locks': False, u'sextette': False, u'chin': False, u'admires': False, u'admirer': False, u'succumbing': False, u'chia': False, u'occur': False, u'listens': False, u'gentler': False, u'savoy': False, u'septic': False, u'discussion': False, u'spreads': False, u'vainly': False, u'cubism': False, u'thanking': False, u'edouard': False, u'klumps': False, u'strays': False, u'rewatched': False, u'deteriorate': False, u'armies': False, u'mintues': False, u'mat': False, u'alger': False, u'pesimism': False, u'casualness': False, u'ornery': False, u'mythos': False, u'convincingly': False, u'islamic': False, u'meddled': False, u'produce': False, u'thurgood': False, u'heterosexuals': False, u'horks': False, u'drastic': False, u'brainless': False, u'egotistical': False, u'surfing': False, u'jonnie': False, u'grandson': False, u'conscious': False, u'tuptim': False, u'irwin': False, u'berets': False, u'amercian': False, u'regressive': False, u'nebbish': False, u'skirmish': False, u'laff': False, u'wolves': False, u'pulled': False, u'manga': False, u'impactful': False, u'suddeth': False, u'serving': False, u'twitchy': False, u'barnyard': False, u'toward': False, u'years': True, u'professors': False, u'structuring': False, u'episodes': False, u'affirms': False, u'marshmallows': False, u'professory': False, u'still': True, u'irene': False, u'sigourney': False, u'overlord': False, u'disconnect': False, u'slimeball': False, u'jia': False, u'milked': False, u'jim': False, u'troubles': False, u'pulitzer': False, u'rudnick': False, u'roadster': False, u'jip': False, u'suspension': False, u'troubled': False, u'tiara': False, u'rationality': False, u'emigrants': False, u'correspondence': False, u'modestly': False, u'non': False, u'sparing': False, u'broadsword': False, u'recipients': False, u'civilian': False, u'nod': False, u'reject': False, u'courageously': False, u'indigenous': False, u'overpowering': False, u'drilling': False, u'base': False, u'workmanlike': False, u'henpecked': False, u'sorted': False, u'nov': False, u'now': True, u'\\': False, u'nor': False, u'josh': False, u'inversion': False, u'placate': False, u'undermining': False, u'materialized': False, u'didn': True, u'didi': False, u'blockheaded': False, u'dispite': False, u'fisherman': False, u'ellicit': False, u'hauff': False, u'battleships': False, u'instability': False, u'quarter': False, u'quartet': False, u'materializes': False, u'consummated': False, u'retrieve': False, u'policed': False, u'bursting': False, u'challenged': False, u'receipt': False, u'crossan': False, u'yeah': False, u'hitmen': False, u'challenges': False, u'challenger': False, u'replay': False, u'sponsor': False, u'entering': False, u'year': False, u'naming': False, u'buzzcocks': False, u'salads': False, u'disasters': False, u'tried': False, u'bouyant': False, u'funnest': False, u'rioters': False, u'interned': False, u'yojimbo': False, u'1992': False, u'thirst': False, u'wiseguys': False, u'disaster_': False, u'seriously': False, u'wholeheartedly': False, u'trauma': False, u'firorina': False, u'internet': False, u'pharoah': False, u'merpeople': False, u'ladder': False, u'igniting': False, u'charlotte': False, u'rebuilding': False, u'complicates': False, u'disintegrated': False, u'hairdresser': False, u'sympathize': False, u'odoriferous': False, u'divergence': False, u'restless': False, u'existentialist': False, u'trendily': False, u'complicated': False, u'mcferran': False, u'grandma': False, u'gaffe': False, u'dlose': False, u'appealingly': False, u'marla': False, u'eggar': False, u'contemplation': False, u'bared': False, u'tomahawk': False, u'tasting': False, u'modest': False, u'marlo': False, u'initiate': False, u'tangled': False, u'aboard': False, u'socking': False, u'ziembicki': False, u'domed': False, u'neglect': False, u'emotion': False, u'gunshot': False, u'romania': False, u'saving': False, u'achingly': False, u'symmetry': False, u'spoken': False, u'velda': False, u'savini': False, u'westlake': False, u'flipping': False, u'reprisal': False, u'one': True, u'_not_': False, u'respecting': False, u'ony': False, u'punishable': False, u'periodical': False, u'haviland': False, u'tamara': False, u'tame': False, u'rina': False, u'onw': False, u'plotless': False, u'exaggerations': False, u'stifler': False, u'stifles': False, u'jugs': False, u'tenko': False, u'thankless': False, u'burst': False, u'remotely': False, u'davidovitch': False, u'hummm': False, u'stifled': False, u'backwoods': False, u'dunces': False, u'megalomaniac': False, u'oversimplified': False, u'fo': False, u'lingering': False, u'featherbrained': False, u'brainy': False, u'beesley': False, u'cherbourg': False, u'uninformed': False, u'shawn': False, u'brains': False, u'appreciates': False, u'surges': False, u'obtained': False, u'snatch': False, u'devito': False, u'1500s': False, u'anthesis': False, u'appearence': False, u'absorbs': False, u'thai': False, u'smirkingly': False, u'heretical': False, u'appreciated': False, u'padre': False, u'professionals': False, u'underwritten': False, u'rza': False, u'hoyle': False, u'gisbourne': False, u'transferred': False, u'crossroads': False, u'admitedly': False, u'rehab': False, u'shyamalan': False, u'vocals': False, u'cymbals': False, u'actioners': False, u'wandering': False, u'fruity': False, u'disasterous': False, u'dilemnas': False, u'bulow': False, u'illness': False, u'aaaaaaaahhhh': False, u'stylings': False, u'overdosing': False, u'sumptuous': False, u'premier': False, u'turned': False, u'locations': False, u'jewels': False, u'bugs': False, u'balsan': False, u'odile': False, u'uninterrupted': False, u'infomercial': False, u'breakfast': False, u'emmylou': False, u'politicos': False, u'jacks': False, u'snorting': False, u'pimply': False, u'shriker': False, u'zoe': False, u'hoosiers': False, u'cigarettes': False, u'warriors': False, u'reasonable': False, u'zoo': False, u'equipment': False, u'goodman': False, u'portents': False, u'martineau': False, u'microphage': False, u'_titus_andronicus_': False, u'mayer': False, u'pimple': False, u'topping': False, u'opposite': False, u'attractiveness': False, u'discerning': False, u'neatly': False, u'spewing': False, u'suspects': False, u'buffet': False, u'intergalactic': False, u'importantly': False, u'printed': False, u'knowingly': False, u'wittliff': False, u'buffed': False, u'parlays': False, u'captivatingly': False, u'wacked': False, u'touchy': False, u'phil': False, u'implications': False, u'premiered': False, u'mrs': False, u'toucha': False, u'jitters': False, u'messier': False, u'wreaked': False, u'slaver': False, u'jittery': False, u'atlantic': False, u'steadi': False, u'rozema': False, u'delroy': False, u'wynn': False, u'discoverer': False, u'fakeouts': False, u'imagines': False, u'zanetti': False, u'friction': False, u'fecal': False, u'oderkerk': False, u'inconsistent': False, u'teamed': False, u'copyrighted': False, u'perfekt': False, u'soviets': False, u'_andre_': False, u'imagined': False, u'unloaded': False, u'wynt': False, u'geography': False, u'renaldi': False, u'snack': False, u'collegue': False, u'zahn': False, u'reconciling': False, u'nicoletta': False, u'nicolette': False, u'embittered': False, u'coaxing': False, u'goregeous': False, u'preternatural': False, u'enriched': False, u'chopin': False, u'sierra': False, u'stain': False, u'shrill': False, u'fades': False, u'guarded': False, u'rejoiced': False, u'granddaddy': False, u'suitcases': False, u'revolutionized': False, u'tilting': False, u'duel': False, u'leguizimo': False, u'undetected': False, u'simplistic': False, u'truffaut': False, u'isis': False, u'awaiting': False, u'miming': False, u'unanimous': False, u'wahlberg': False, u'blind': False, u'deconstruction': False, u'pimp': False, u'lovebirds': False, u'trys': False, u'carrion': False, u'ambling': False, u'recoiling': False, u'musetta': False, u'choudhury': False, u'vision': False, u'morose': False, u'osbourne': False, u'attenuated': False, u'incarcerated': False, u'childbearings': False, u'underbids': False, u'audaciously': False, u'concubines': False, u'senitmental': False, u'actively': False, u'impressions': False, u'inseminates': False, u'clarinet': False, u'intoxicating': False, u'aboslutely': False, u'defensively': False, u'retells': False, u'masturbatory': False, u'libidinous': False, u'alarming': False, u'feuds': False, u'glaze': False, u'heartstrings': False, u'rollicking': False, u'sponsorship': False, u'thrillerism': False, u'_dirty_work_': False, u'altercation': False, u'moons': False, u'nicest': False, u'enjoys': False, u'playhouse': False, u'orchestrate': False, u'caan': False, u'tsui': False, u'lyricized': False, u'faded': False, u'braggarts': False, u'faking': False, u'punts': False, u'awards': False, u'hurts': False, u'menacing': False, u'moans': False, u'innuendos': False, u'smoggy': False, u'uncharacteristically': False, u'concentrated': False, u'busting': False, u'yorick': False, u'confection': False, u'majestically': False, u'resides': False, u'rhodes': False, u'matheson': False, u'millionaire': False, u'flipped': False, u'prompted': False, u's': True, u'workplace': False, u'yammering': False, u'concentrates': False, u'flipper': False, u'doctoring': False, u'loveliest': False, u'madness': False, u'beowolf': False, u'adroit': False, u'foreboding': False, u'sixth': False, u'jeans': False, u'inexplicable': False, u'marriageable': False, u'inexplicably': False, u'152': False, u'mugshots': False, u'ragtag': False, u'imbues': False, u'brats': False, u'comparitive': False, u'undercuts': False, u'collides': False, u'gynecomastia': False, u'west': False, u'deuteronomy': False, u'tropical': False, u'collided': False, u'dictator': False, u'brancia': False, u'motives': False, u'partisanship': False, u'sked': False, u'nntphub': False, u'spyglass': False, u'odessa': False, u'wants': False, u'dussander': False, u'straying': False, u'vomits': False, u'tomei': False, u'elvis': False, u'formed': False, u'photon': False, u'readings': False, u'photos': False, u'tightened': False, u'goran': False, u'abject': False, u'former': False, u'sedition': False, u'sommers': False, u'dostoevsky': False, u'chauvinistic': False, u'winks': False, u'defeatist': False, u'straighten': False, u'squeezes': False, u'shockwave': False, u'diverse': False, u'newspaper': False, u'situation': False, u'slapping': False, u'prevue': False, u'penthouse': False, u'unlikeable': False, u'spungen': False, u'rapier': False, u'mucus': False, u'inflatable': False, u'surveying': False, u'engaged': False, u'zucker': False, u'dubious': False, u'_still_': False, u'menancing': False, u'92ve': False, u'twotg': False, u'korben': False, u'waterway': False, u'engages': False, u'multitudes': False, u'misanthropy': False, u'nile': False, u'debilitating': False, u'ingrained': False, u'nuptials': False, u'fistfights': False, u'gediman': False, u'amateurs': False, u'quagmire': False, u'quiclky': False, u'otto': False, u'jessalyn': False, u'bogglingly': False, u'adolf': False, u'reprintable': False, u'visually': False, u'wires': False, u'edged': False, u'assigns': False, u'hideaway': False, u'sickness': False, u'krippendorf': False, u'defy': False, u'brassed': False, u'dubya': False, u'deflate': False, u'burkhart': False, u'tolan': False, u'edges': False, u'amuck': False, u'plops': False, u'advertisement': False, u'ratttz': False, u'nandita': False, u'_seven_nights_': False, u'carhart': False, u'tracking': False, u'droppingly': False, u'eventual': False, u'irresistable': False, u'charges': False, u'bagpipes': False, u'organs': False, u'sorderbergh': False, u'nothin': False, u'snarfed': False, u'peculiarities': False, u'demean': False, u'alleviate': False, u'delectably': False, u'penetration': False, u'dimension': False, u'persistently': False, u'taxi': False, u'livestock': False, u'recycles': False, u'being': False, u'battleship': False, u'neon': False, u'moviestar': False, u'bueller': False, u'recycled': False, u'compromising': False, u'maternity': False, u'senatorial': False, u'dick_': False, u'parlay': False, u'lonesome': False, u'procreating': False, u'rover': False, u'renewed': False, u'grounded': False, u'cloris': False, u'lifelong': False, u'rescored': False, u'verse': False, u'serous': False, u'gloating': False, u'versa': False, u'overthrow': False, u'haystack': False, u'dicks': False, u'ballistics': False, u'kareem': False, u'ami': False, u'absense': False, u'phelps': False, u'onscreen': False, u'sportsmanship': False, u'rejoin': False, u'amc': False, u'sums': False, u'gonna': False, u'unveil': False, u'sumo': False, u'earnest': False, u'permanence': False, u'traffic': False, u'preference': False, u'lads': False, u'ramshackle': False, u'world': True, u'embrassment': False, u'postal': False, u'reap': False, u'likeablity': False, u'sensational': False, u'malfunctions': False, u'fortune': False, u'heightened': False, u'unrepentant': False, u'cheerily': False, u'unrequited': False, u'conducts': False, u'dynasties': False, u'yearnings': False, u'benefit': False, u'superiority': False, u'glamor': False, u'output': False, u'dirtier': False, u'petrice': False, u'confrontatory': False, u'twelve': False, u'satisfactory': False, u'superintendent': False, u'verbal': False, u'affay': False, u'sentimentally': False, u'dilbert': False, u'tragedies': False, u'tvs': False, u'exposes': False, u'stewardess': False, u'magma': False, u'zealously': False, u'demeaning': False, u'diving': False, u'stagecoach': False, u'divine': False, u'bongos': False, u'derisive': False, u'regina': False, u'dancefloor': False, u'painstakingly': False, u'soooo': False, u'bottlecaps': False, u'cavity': False, u'shakedown': False, u'seaman': False, u'charismatic': False, u'babysitter': False, u'garret': False, u'guillaume': False, u'squirt': False, u'francois': False, u'911': False, u'restoring': False, u'predators': False, u'fractures': False, u'muttered': False, u'process': False, u'squabble': False, u'tomorrow': False, u'arrow': True, u'macgowan': False, u'retains': False, u'cliquey': False, u'tv2': False, u'aquit': False, u'leadership': False, u'innocence': False, u'piscapo': False, u'thailand': False, u'improvisation': False, u'fairy': False, u'demarco': False, u'exasperating': False, u'loyalties': False, u'creaking': False, u'jungian': False, u'piety': False, u'hopkins': False, u'majorino': False, u'ob': False, u'stanleyville': False, u'ideology': False, u'wrestled': False, u'mugging': False, u'frights': False, u'niall': False, u'chow': False, u'internalize': False, u'johnston': False, u'carribean': False, u'backup': False, u'locklear': False, u'sensitively': False, u'rabbinical': False, u'freeing': False, u'perturbed': False, u'shapely': False, u'burial': False, u'antidote': False, u'kroon': False, u'cleaves': False, u'ineffable': False, u'lively': False, u'bukater': False, u'pivot': False, u'star_': False, u'conceptually': False, u'shrinking': False, u'rossi': False, u'uhhhhhm': False, u'complexly': False, u'distractedness': False, u'vaporize': False, u'shags': False, u'hunks': False, u'gleam': False, u'glean': False, u'stark': False, u'lounging': False, u'redirection': False, u'mindless': False, u'missy': False, u'sealed': False, u'brazilian': False, u'bubble': False, u'allergic': False, u'witt': False, u'lasseter': False, u'continents': False, u'maltin': False, u'wits': False, u'smuggling': False, u'bohemians': False, u'pitcher': False, u'lane': False, u'societal': False, u'philosophers': False, u'foreheads': False, u'hudgeons': False, u'recouped': False, u'omelet': False, u'attainable': False, u'pitched': False, u'with': True, u'abused': False, u'pull': False, u'rush': False, u'thumps': False, u'illustrator': False, u'dominican': False, u'embedded': False, u'rage': False, u'default': False, u'tripe': False, u'claustral': False, u'chomped': False, u'freddy': False, u'rags': False, u'waltzing': False, u'dirty': False, u'abuser': False, u'abuses': False, u'russ': False, u'trips': False, u'touchstone': False, u'mcgruder': False, u'patois': False, u'607': False, u'falseness': False, u'wormwood': False, u'gratuitous': False, u'transporters': False, u'macbeth': False, u'watches': False, u'watcher': False, u'stigmata': False, u'associating': False, u'elmaloglou': False, u'immobilise': False, u'toontown': False, u'watched': False, u'jargon': False, u'tremble': False, u'dampens': False, u'cream': False, u'cabinets': False, u'moniker': False, u'ideally': False, u'administered': False, u'yogi': False, u'sympathetically': False, u'unwelcomed': False, u'introspection': False, u'hofstra': False, u'unparalleled': False, u'friggin': False, u'fools': False, u'puppy': False, u'poor': False, u'poop': False, u'treks': False, u'diaries': False, u'cazale': False, u'endeavors': False, u'addictions': False, u'whistling': False, u'artillary': False, u'waving': False, u'falstaff': False, u'midget': False, u'brotherhood': False, u'whippersnappers': False, u'torches': False, u'linebackers': False, u'poon': False, u'fedoras': False, u'pool': False, u'tricky': False, u'mourned': False, u'titillating': False, u'natalie': False, u'thora': False, u'tricks': False, u'maliciously': False, u'smooch': False, u'dyed': False, u'uploaded': False, u'moonlight': False, u'corey': False, u'satirize': False, u'finklestein': False, u'dyer': False, u'overseas': False, u'vadis': False, u'humoring': False, u'sci': False, u'anette': False, u'lopped': False, u'caused': False, u'beware': False, u'month': False, u'slimming': False, u'zappa': False, u'thoughtful': False, u'concept': True, u'upholds': False, u'greenwood': False, u'suspected': False, u'criticising': False, u'jolly': False, u'religious': False, u'causes': False, u'corps': False, u'pledged': False, u'wyoming': False, u'riots': False, u'nora': False, u'danielle': False, u'undiscernable': False, u'safety': False, u'conciousness': False, u'fluently': False, u'carters': False, u'norm': False, u'gazarra': False, u'oleynik': False, u'deedle': False, u'resounding': False, u'patently': False, u'7': True, u'clubbed': False, u'powaqqatsi': False, u'horror': True, u'floated': False, u'coloured': False, u'clubber': False, u'decide': False, u'24th': False, u'suspensefully': False, u'sant': False, u'rootless': False, u'multihued': False, u'sans': False, u'boozed': False, u'shenanigans': False, u'ceaseless': False, u'unfailingly': False, u'sang': False, u'sand': False, u'sane': False, u'saboteur': False, u'unwraps': False, u'small': False, u'anal': False, u'sank': False, u'realisticly': False, u'vanquish': False, u'doofy': False, u'courtesans': False, u'abbreviated': False, u'quicker': False, u'lulls': False, u'traditions': False, u'streets': False, u'tardis': False, u'healed': False, u'past': False, u'orgasm': False, u'asp': False, u'neuromancer': False, u'displays': False, u'notting': False, u'pass': False, u'healer': False, u'befriends': False, u'marginal': False, u'investment': False, u'chuckle': False, u'amarcord': False, u'offbeat': False, u'cues': False, u'clock': False, u'investing': False, u'zealots': False, u'skywalker': False, u'lurch': False, u'leit': False, u'colonists': False, u'bernhard': False, u'learned': False, u'leia': False, u'psychoanalysts': False, u'dwells': False, u'hasn': False, u'full': False, u'hash': False, u'sabara': False, u'diapers': False, u'portrays': False, u'scum': False, u'tracks': False, u'bribes': False, u'civilians': False, u'eventful': False, u'november': False, u'hass': False, u'melancholic': False, u'houser': False, u'contrastingly': False, u'ivey': False, u'conviction': False, u'inspired': False, u'losses': False, u'experience': False, u'anthropologists': False, u'prior': False, u'beaman': False, u'periodic': False, u'holdover': False, u'_all_': False, u'bitchie': False, u'cessation': False, u'divison': False, u'skepticism': False, u'hime': False, u'inspires': False, u'amadeus': False, u'uniquely': False, u'interactivity': False, u'norville': False, u'followed': False, u'retroactive': False, u'mediator': False, u'scharzenegger': False, u'returing': False, u'traumatized': False, u'straightened': False, u'follower': False, u'asexually': False, u'analyzing': False, u'gigs': False, u'traumatizes': False, u'dispondent': False, u'cynics': False, u'seventeenth': False, u'enlightened': False, u'automats': False, u'volcanos': False, u'silva': False, u'attendance': False, u'nehru': False, u'enliven': False, u'ladybugs': False, u'canoe': False, u'briesewitz': False, u'lollipop': False, u'mori': False, u'unrecognizable': False, u'certified': False, u'restraints': False, u'firth': False, u'mora': False, u'gaetan': False, u'glowers': False, u'more': True, u'cannibalism': False, u'lowering': False, u'israel': False, u'barbarino': False, u'door': False, u'doos': False, u'initiated': False, u'chucky': False, u'fairman': False, u'company': False, u'corrected': False, u'dishwasher': False, u'tested': False, u'lameness': False, u'fumble': False, u'7000': False, u'lazard': False, u'doom': False, u'producing': False, u'alana': False, u'negativity': False, u'hotchner': False, u'leary': False, u'kaminski': False, u'fornicators': False, u'pollack': False, u'mccracken': False, u'maniac': False, u'patriarch': False, u'kaminsky': False, u'budgets': False, u'learn': False, u'knocked': False, u'infectuous': False, u'grope': False, u'scramble': False, u'barclay': False, u'allegra': False, u'casanova': False, u'bogs': False, u'memoirs': False, u'rejuvenate': False, u'meaner': False, u'publicist': False, u'barzoon': False, u'freebie': False, u'bogg': False, u'aatish': False, u'prostration': False, u'sikh': False, u'huge': False, u'respective': False, u'hickey': False, u'edgecomb': False, u'benji': False, u'demolition': False, u'speedboat': False, u'hugo': False, u'hugh': False, u'ledger': False, u'dismissed': False, u'surpassed': False, u'snippet': False, u'dismembering': False, u'veloz': False, u'benzali': False, u'hugs': False, u'surpasses': False, u'dismisses': False, u'isuro': False, u'schultz': False, u'espousing': False, u'sprinkle': False, u'lanky': False, u'intended': False, u'mendes': False, u'risqu': False, u'thickened': False, u'disgraced': False, u'greenwald': False, u'fa': False, u'unsung': False, u'fc': False, u'hackwork': False, u'seafaring': False, u'maltese': False, u'fi': False, u'horizontal': False, u'dryland': False, u'compulsory': False, u'malevolent': False, u'fairfolk': False, u'midsection': False, u'criticize': False, u'jiang': False, u'relocating': False, u'resemble': False, u'excitment': False, u'sublte': False, u'twisting': False, u'tiegs': False, u'anytime': False, u'house_': False, u'dolph': False, u'renewing': False, u'daggers': False, u'roommates': False, u'chopsticks': False, u'overcooked': False, u'semantic': False, u'grizzled': False, u'replied': False, u'unexpectantly': False, u'weirdoes': False, u'remnants': False, u'casablanca': False, u'depraved': False, u'peppy': False, u'bruskotter': False, u'installed': False, u'resorts': False, u'thermal': False, u'paper': False, u'scott': False, u'signs': False, u'booboos': False, u'smiling': False, u'signy': False, u'roots': False, u'saucy': False, u'mistreated': False, u'tantalizingly': False, u'ethnocentric': False, u'picard': False, u'sublimated': False, u'overexcited': False, u'hounds': False, u'isaak': False, u'absence': False, u'blunderheaded': False, u'dolly': False, u'bummer': False, u'isaac': False, u'curiousity': False, u'cutie': False, u'sauce': False, u'reintroduced': False, u'colleague': False, u'cartman': False, u'pleads': False, u'frizzi': False, u'abandons': False, u'motherless': False, u'universality': False, u'gadget': False, u'furthermore': False, u'awarding': False, u'musker': False, u'frizzy': False, u'starbucks': False, u'balaban': False, u'weeds': False, u'idols': False, u'ninja': False, u'petter': False, u'burkhard': False, u'slighted': False, u'prayed': False, u'bryan': False, u'faire': False, u'everytime': False, u'veering': False, u'denny': False, u'durden': False, u'courses': False, u'unbrewed': False, u'hatchette': False, u'repayment': False, u'turns': False, u'shocking': False, u'obligated': False, u'lifestyles': False, u'predatory': False, u'flashbacks': False, u'reactions': False, u'brunette': False, u'another': False, u'jaws': False, u'exasperated': False, u'tamiyo': False, u'numeric': False, u'scalvaging': False, u'chrissy': False, u'gum': False, u'squishy': False, u'operation': False, u'puppeteer': False, u'absurdity': False, u'anarchy': False, u'centuries': False, u'inquired': False, u'unsurpassed': False, u'marcie': False, u'lipstick': False, u'ernie': False, u'marcia': False, u'visions': True, u'malcontents': False, u'kensington': False, u'buzzsaw': False, u'research': False, u'fountain': False, u'inquires': False, u'enigmatical': False, u'illustrate': False, u'hazardus': False, u'occurs': False, u'foam': False, u'earns': False, u'dunn': False, u'housed': False, u'favored': False, u'toiling': False, u'stores': False, u'pub': False, u'factual': False, u'houses': False, u'johnnie': False, u'abnormally': False, u'hedaya': False, u'poignantly': False, u'put': False, u'hapless': False, u'speculation': False, u'bash': False, u'definition': False, u'pairs': False, u'unresolved': False, u'2056': False, u'americas': False, u'sheikh': False, u'cheering': False, u'american': True, u'genuises': False, u'theroux': False, u'unjustifyably': False, u'authoritive': False, u'testament': False, u'existential': False, u'matarazzo': False, u'apocalypse': False, u'porpoise': False, u'petra': False, u'horndog': False, u'seduction': False, u'trekkie': False, u'terrifyingly': False, u'encoding': False, u'watchdogs': False, u'brutally': False, u'preservation': False, u'moritz': False, u'elder': False, u'morita': False, u'craps': False, u'effortless': False, u'intellectualism': False, u'miss': False, u'financing': False, u'heralds': False, u'burke': False, u'equating': False, u'horse': False, u'eightly': False, u'bakula': False, u'blossom': False, u'arlington': False, u'nomi': False, u'moderately': False, u'heartedness': False, u'bigscreen': False, u'excitable': False, u'bedridden': False, u'station': False, u'shamed': False, u'saint': False, u'kindergartner': False, u'essays': False, u'hundred': False, u'merciless': False, u'peaceably': False, u'nursery': False, u'justly': False, u'feels': True, u'pooh': False, u'trapped': False, u'dethroned': False, u'interviewed': False, u'1919': False, u'trapper': False, u'typhoon': False, u'cheekbones': False, u'chapelle': False, u'theater': False, u'1913': False, u'stifle': False, u'oilrig': False, u'refilmed': False, u'dethrones': False, u'pyroclastic': False, u'funicello': False, u'stormare': False, u'condolences': False, u'automobiles': False, u'bruckheimer': False, u'getaway': False, u'hesitating': False, u'dogs': False, u'stu': False, u'swanbeck': False, u'dismantling': False, u'cuff': False, u'beneficial': False, u'prescott': False, u'labyrinthian': False, u'populating': False, u'navigators': False, u'breakneck': False, u'phobia': False, u'kret': False, u'contraception': False, u'mensch': False, u'tearfully': False, u'conventional': False, u'organisations': False, u'swanky': False, u'waft': False, u'berle': False, u'guarding': False, u'graffiti': False, u'blond': False, u'cleon': False, u'cleverness': False, u'disilusioned': False, u'sell': False, u'nosebleeding': False, u'antagonizes': False, u'foot': False, u'blanco': False, u'tarnish': False, u'self': False, u'cave': False, u'sela': False, u'client': False, u'also': True, u'recognizing': False, u'sebastiano': False, u'conscription': False, u'sharpe': False, u'bastad': False, u'vomitted': False, u'pringles': False, u'offscreen': False, u'caliber': False, u'hires': False, u'_election': False, u'rapist': False, u'singles': False, u'warhol': False, u'systems': False, u'wallpaperer': False, u'raucous': False, u'virus': False, u'hired': False, u'channeling': False, u'unbeknownst': False, u'immediacy': False, u'singled': False, u'hypotheitically': False, u'understands': False, u'raving': False, u'omaha': False, u'ballyhoo': False, u'seize': False, u'mottled': False, u'sometimes': False, u'rockefeller': False, u'humbert': False, u'flits': False, u'barred': False, u'lounge': False, u'cultivating': False, u'barren': False, u'figurines': False, u'barrel': False, u'shread': False, u'amusements': False, u'takashi': False, u'dragonflies': False, u'ambiguities': False, u'stalked': False, u'ugh': False, u'phenomenon': False, u'ugc': False, u'blended': False, u'accommodations': False, u'foregoing': False, u'netherworld': False, u'colorized': False, u'serialised': False, u'prodigious': False, u'naomi': False, u'keital': False, u'heavens': False, u'overwhelmed': False, u'caruso': False, u'distressing': False, u'wraps': False, u'1912': False, u'kinkiness': False, u'neophytes': False, u'turmoil': False, u'indiglo': False, u'gooey': False, u'1914': False, u'cassette': False, u'snobbish': False, u'crashlands': False, u'1916': False, u'indifference': False, u'lombard': False, u'tarkofsky': False, u'haskin': False, u'secular': False, u'ceasing': False, u'sunny': False, u'asssss': False, u'informants': False, u'yeager': False, u'remedy': False, u'assaulted': False, u'_breakfast_of_champions_': False, u'pretending': False, u'compass': False, u'boils': False, u'damnit': False, u'_54_': False, u'distraction': False, u'===========': False, u'adlib': False, u'devoured': False, u'sects': False, u'pleasures': False, u'stroll': False, u'tanked': False, u'sojourn': False, u'stored': False, u'miseries': False, u'hype': False, u'forster': False, u'tanker': False, u'pleasured': False, u'rumored': False, u'insane': False, u'delicately': False, u'howled': False, u'patriotism': False, u'bozo': False, u'activists': False, u'grey': False, u'collectively': False, u'howler': False, u'overboard': False, u'portrait': False, u'gouda': False, u'allie': False, u'glistening': False, u'termites': False, u'colonialists': False, u'richelieu': False, u'commensurately': False, u'ahmet': False, u'tolls': False, u'storyboarded': False, u'greg': False, u'monroth': False, u'falsehood': False, u'honour': False, u'wopr': False, u'15th': False, u'goggins': False, u'zippel': False, u'missing': False, u'handyman': False, u'zipped': False, u'untouched': False, u'spray': False, u'ranked': False, u'coffee': False, u'canran': False, u'zipper': False, u'hairpin': False, u'abruptly': False, u'hoisted': False, u'collaborators': False, u'lass': False, u'last': False, u'legitimately': False, u'cashing': False, u'opal': False, u'swigert': False, u'heros': False, u'rehired': False, u'connection': False, u'amoeba': False, u'opar': False, u'retarded': False, u'lash': False, u'vault': False, u'audiotapes': False, u'experimental': False, u'onofrio': False, u'coifed': False, u'expendable': False, u'bell': False, u'2293': False, u'acted': False, u'america': False, u'ffing': False, u'unfolded': False, u'adaptation': False, u'seldes': False, u'jitterish': False, u'belt': False, u'unthrilling': False, u'warfield': False, u'eldard': False, u'unarguably': False, u'satire': False, u'suburbs': False, u'proprietor': False, u'initiation': False, u'portait': False, u'faulkner': False, u'patrolled': False, u'combatants': False, u'crowns': False, u'sending': False, u'infect': False, u'exposed': False, u'amphibians': False, u'adaptable': False, u'awake': False, u'organa': False, u'mournful': False, u'schwarztman': False, u'randomly': False, u'magwitch': False, u'exponential': False, u'caged': False, u'expanded': False, u'budget': False, u'admire': False, u'reopens': False, u'cagey': False, u'pressed': False, u'frighteners': False, u'bogan': False, u'cages': False, u'beatng': False, u'exclude': False, u'voe': False, u'agitation': False, u'mystic': False, u'von': False, u'binding': False, u'faceted': False, u'sawalha': False, u'vow': False, u'boredom': False, u'blissful': False, u'underlining': False, u'cuisine': False, u'implacable': False, u'raiders': False, u'stunk': False, u'jerking': False, u'perpetrator': False, u'burdensome': False, u'pridefully': False, u'matchmakers': False, u'cayman': False, u'ugliness': False, u'bangs': False, u'windmill': False, u'praising': False, u'flooded': False, u'katana': False, u'everclear': False, u'reiser': False, u'fleischer': False, u'wunderkind': False, u'adrift': False, u'ditz': False, u'vargas': False, u'infamous': False, u'symbolise': False, u'doreen': False, u'prehensile': False, u'coachmen': False, u'dared': False, u'portillo': False, u'bangy': False, u'scoffs': False, u'thats': False, u'hammiest': False, u'soaked': False, u'pepto': False, u'ceo': False, u'salva': False, u'cheddar': False, u'crackled': False, u'thaddeus': False, u'hercules': False, u'stunt': False, u'crackles': False, u'uzi': False, u'wage': False, u'hemingwayesque': False, u'cuffs': False, u'melinda': False, u'kurt': False, u'rangers': False, u'studious': False, u'parents': False, u'depravity': False, u'boardroom': False, u'eery': False, u'cormack': False, u'emergency': False, u'impaling': False, u'couple': False, u'bureaucrat': False, u'emanating': False, u'prayer': False, u'wives': False, u'ofcs': False, u'boost': False, u'abound': False, u'emergence': False, u'curtis': False, u'thurman': False, u'marquee': False, u'spine': False, u'chorus': False, u'individuals': False, u'crookier': False, u'bogie': False, u'mediocrity': False, u'bamboo': False, u'turvy': False, u'alexandre': False, u'spins': False, u'crescendo': False, u'methods': False, u'goddamn': False, u'unsubstantial': False, u'bounce': False, u'ahern': False, u'bouncy': False, u'saintliness': False, u'debello': False, u'greener': False, u'underbelly': False, u'obliges': False, u'measurements': False, u'novelty': False, u'pell': False, u'behave': False, u'disguising': False, u'whodunit': False, u'metamorphoses': False, u'seclusion': False, u'inserting': False, u'dialogueless': False, u'hammond': False, u'jovovich': False, u'wayward': False, u'obscures': False, u'respite': False, u'grotesqe': False, u'janusz': False, u'obscured': False, u'cranked': False, u'deserved': False, u'simplify': False, u'goody': False, u'scorces': False, u'wrinkles': False, u'melbourne': False, u'deserves': False, u'scraggly': False, u'maude': False, u'wrinkled': False, u'gallagher': False, u'canning': False, u'laughton': False, u'aires': False, u'tornatore': False, u'_dead_': False, u'terrorists': False, u'into': True, u'unredeemable': False, u'catchiness': False, u'middleton': False, u'controversies': False, u'remembrance': False, u'chirping': False, u'katie': False, u'realisation': False, u'winnebago': False, u'span': False, u'harnessed': False, u'spam': False, u'meteorological': False, u'tingles': False, u'sock': False, u'overstaying': False, u'gases': False, u'bios': False, u'limburgher': False, u'grave': False, u'mishandle': False, u'spar': False, u'purred': False, u'spat': False, u'considerably': False, u'atlantis': False, u'invite': False, u'hawaiian': False, u'murphy': False, u'palentologist': False, u'_dragon_': False, u'deductions': False, u'lydia': False, u'carping': False, u'fanatasies': False, u'considerable': False, u'intestines': False, u'jacki': False, u'remastered': False, u'charmed': False, u'erich': False, u'pissant': False, u'testaments': False, u'sturdy': False, u'eddie': False, u'erica': False, u'paired': False, u'kinsella': False, u'tamahori': False, u'intently': False, u'kihlstedt': False, u'awestruck': False, u'chad': False, u'pseudonymous': False, u'influence': False, u'haunt': False, u'arena': False, u'portentuous': False, u'globally': False, u'thomsen': False, u'chap': False, u'revelatory': False, u'palisades': False, u'chat': False, u'apropos': False, u'chaz': False, u'frontgate': False, u'immeadiately': False, u'neilsen': False, u'cathy': False, u'intrepid': False, u'puzzling': False, u'copulate': False, u'thanks': False, u'ginty': False, u'excuses': False, u'conceptions': False, u'ellen': False, u'singed': False, u'heebie': False, u'aunt': False, u'rabal': False, u'interogation': False, u'oblige': False, u'gardenia': False, u'teck': False, u'strums': False, u'aussies': False, u'prepared': False, u'bianca': False, u'mckidd': False, u'manipulator': False, u'flyboy': False, u'suppression': False, u'velocity': False, u'wedlock': False, u'euphegenia': False, u'mckenney': False, u'lang': False, u'guitry': False, u'land': False, u'ryan_': False, u'lana': False, u'physics': False, u'unlocked': False, u'advertisment': False, u'purged': False, u'jansen': False, u'reserve': False, u'modernizing': False, u'contortions': False, u'zzzzzzz': False, u'splashing': False, u'spielbergization': False, u'unbuttoning': False, u'broader': False, u'amiss': False, u'flashback': False, u'humpback': False, u'coffey': False, u'detectives': False, u'amalgamation': False, u'turkish': False, u'ditzism': False, u'horsing': False, u'dickinson': False, u'carelessly': False, u'resources': False, u'nervousness': False, u'lindner': False, u'boatload': False, u'undeterred': False, u'millieu': False, u'alienbusting': False, u'hockley': False, u'huddled': False, u'prakazrel': False, u'traumatised': False, u'scatology': False, u'koppelman': False, u'petitions': False, u'decorating': False, u'herzfeld': False, u'detested': False, u'yakov': False, u'lombardo': False, u'rifling': False, u'integrating': False, u'ineffectuality': False, u'fewer': False, u'damning': False, u'yevgeny': False, u'sarandon': False, u'disheveled': False, u'insubordinate': False, u'leonardi': False, u'leonardo': False, u'villiany': False, u'maclachlan': False, u'overblown': False, u'dysfuntion': False, u'cannibals': False, u'stalker': False, u'mishap': False, u'crook': False, u'video': True, u'=====================': False, u'dynamics': False, u'elisa': False, u'victor': False, u'improvisationaly': False, u'narrations': False, u'sweats': False, u'waning': False, u'harvests': False, u'pledges': False, u'sweaty': False, u'henceforth': False, u'royalist': False, u'turnaround': False, u'slickster': False, u'flowing': False, u'charade': False, u'harassing': False, u'guamo': False, u'forwarned': False, u'apace': False, u'francie': False, u'squirming': False, u'fifteen': False, u'implicit': False, u'kriss': False, u'bakersfield': False, u'33': False, u'unwittingly': False, u'scatter': False, u'condescending': False, u'panned': False, u'survey': False, u'climb': False, u'makes': True, u'maker': False, u'looted': False, u'bumming': False, u'panicked': False, u'blizzard': False, u'assemble': False, u'formulates': False, u'dumbest': False, u'chilly': False, u'desiring': False, u'confidence': False, u'francis': False, u'excising': False, u'pfarrer': False, u'gregor': False, u'zsigmond': False, u'next': False, u'eleven': False, u'assuring': False, u'mccleod': False, u'chu': False, u'tahoe': False, u'gales': False, u'binges': False, u'start': True, u'phallus': False, u'yugoslavian': False, u'pencil': False, u'babe': False, u'spearing': False, u'aphrodite': False, u'tons': True, u'duper': False, u'babs': False, u'boondocks': False, u'prizes': False, u'losin': False, u'baby': False, u'antichrist': False, u'_escape': False, u'documentarian': False, u'customer': False, u'rotterdam': False, u'f': False, u'clients': False, u'attachs': False, u'unknowns': False, u'boorish': False, u'retell': False, u'harve': False, u'initation': False, u'rehabilitation': False, u'wedge': False, u'loca': False, u'painkiller': False, u'calculation': False, u'lock': False, u'coolness': False, u'loco': False, u'promotional': False, u'aughra': False, u'nears': False, u'chic': False, u'bolstered': False, u'taj': False, u'cecilia': False, u'educational': False, u'afi': False, u'raeeyain': False, u'awkwardness': False, u'paled': False, u'schandling': False, u'tightrope': False, u'procured': False, u'neary': False, u'bilingual': False, u'hormones': False, u'burley': False, u'engagingly': False, u'intelligent': False, u'retorts': False, u'pales': False, u'incongruent': False, u'highs': False, u'huffs': False, u'strut': False, u'retrograding': False, u'upstate': False, u'procures': False, u'infirm': False, u'realized': False, u'jolting': False, u'solon': False, u'clarkson': False, u'shout': False, u'robot': False, u'harwich': False, u'realizes': False, u'scrubs': False, u'sciorra': False, u'typicalness': False, u'backus': False, u'marshalls': False, u'bartenders': False, u'houston': False, u'boxing': False, u'thigh': False, u'mute': False, u'muth': False, u'despite': True, u'outlook': False, u'giveaways': False, u'frieberg': False, u'spatula': False, u'directs': False, u'bartusiak': False, u'hotcakes': False, u'perfect': False, u'anonymously': False, u'byline': False, u'rizzo': False, u'jetsons': False, u'noses': False, u'meantime': True, u'thieves': False, u'derivative': False, u'bidets': False, u'90210': False, u'sabotaging': False, u'prosper': False, u'vocalized': False, u'impervious': False, u'overal': False, u'isacsson': False, u'guaspari': False, u'reinvents': False, u'snake': False, u'squabbling': False, u'realize': False, u'reconstruction': False, u'comedy': False, u'damian': False, u'scenic': False, u'zeist': False, u'denzel': False, u'shortage': False, u'brightest': False, u'weismuller': False, u'emo': False, u'glasses': False, u'goldsman': False, u'suitors': False, u'bump': False, u'poppins': False, u'bums': False, u'product': False, u'retaliation': False, u'deficiency': False, u'leplastrier': False, u'books': False, u'resuscitate': False, u'gungan': False, u'bigfoot': False, u'witness': False, u'unoriginal': False, u'matrix': False, u"'": True, u'culture': False, u'harrowingly': False, u'narratively': False, u'escalate': False, u'frowns': False, u'unprepared': False, u'hypnotist': False, u'red': False, u'stare': False, u'unwieldy': False, u'benben': False, u'inferiority': False, u'greedy': False, u'gawain': False, u'disintegrating': False, u'initialize': False, u'disgusted': False, u'pothead': False, u'mainland': False, u'fueled': False, u'blandy': False, u'gallons': False, u'could': False, u'genieveve': False, u'length': False, u'standbys': False, u'qualifying': False, u'chills': False, u'babyzilla': False, u'fleshed': False, u'scene': False, u'reaches': False, u'soothing': False, u'affliction': False, u'leick': False, u'morice': False, u'scent': False, u'fleshes': False, u'braces': False, u'erstwhile': False, u'leder': False, u'festival': False, u'lumet': False, u'locale': False, u'rediscovers': False, u'fanaro': False, u'stabbin': False, u'sergeant': False, u'henning': False, u'pervasive': False, u'enforcement': False, u'zookeeper': False, u'stomach': False, u'stars': False, u'quarry': False, u'greenbaum': False, u'pulman': False, u'beastuality': False, u'incongruities': False, u'fatboy': False, u'defeats': False, u'propagating': False, u'egregious': False, u'roulette': False, u'commandant': False, u'gentile': False, u'orchestrated': False, u'daydreams': False, u'mackey': False, u'faulted': False, u'false': False, u'shrinks': False, u'wakes': False, u'chivalrous': False, u'comedically': False, u'perpetrators': False, u'tonight': False, u'typecasted': False, u'outdoorsman': False, u'ponders': False, u'richman': False, u'sufis': False, u'cecil': False, u'hessian': False, u'depict': False, u'venturing': False, u'dishes': False, u'fireballs': False, u'mia': False, u'dodie': False, u'mib': False, u'nosed': False, u'precipice': False, u'dished': False, u'locals': False, u'bakshi': False, u'sandworms': False, u'worldwide': False, u'jimmies': False, u'closeups': False, u'sol': False, u'manor': False, u'pitches': False, u'fujioka': False, u'petals': False, u'cipher': False, u'draws': False, u'unsexy': False, u'hoodwink': False, u'salutory': False, u'unsparing': False, u'doogie': False, u'auberjonois': False, u'placement': False, u'introversion': False, u'wuthering': False, u'bred': False, u'frederick': False, u'thanksgiving': False, u'lots': False, u'perceiving': False, u'manipulative': False, u'opts': False, u'undersea': False, u'brew': False, u'bret': False, u'}': False, u'jabbar': False, u'rainbows': False, u'scotty': False, u'visualized': False, u'seague': False, u'greenhouse': False, u'drooling': False, u'crows': False, u'thicket': False, u'nominally': False, u'xvi': False, u'taps': False, u'jax': False, u'jay': False, u'jaw': False, u'consciouness': False, u'jar': False, u'corruptor': False, u'risqueness': False, u'gascogne': False, u'jan': False, u'entities': False, u'jam': False, u'tape': False, u'jah': False, u'jai': False, u'riding': False, u'jab': False, u'abbe': False, u'insight': True, u'cooperation': False, u'abba': False, u'antagonism': False, u'prohibition': False, u'molasses': False, u'drawn': False, u'tossed': False, u'macnamara': False, u'wring': False, u'styrofoam': False, u'abby': False, u'brain': False, u'gertz': False, u'rapper': False, u'comprising': False, u'taxes': False, u'shields': False, u'coaxed': False, u'ocmic': False, u'stuff': False, u'pictured': False, u'palotti': False, u'ohio': False, u'rapped': False, u'raceway': False, u'exude': False, u'guessing': False, u'allusion': False, u'qinqin': False, u'affirmations': False, u'frame': False, u'hijinx': False, u'arsed': False, u'kombat_': False, u'alessandro': False, u'trods': False, u'siegfried': False, u'deconstructs': False, u'shallows': False, u'dungeon': False, u'destiny': False, u'insulting': False, u'reform': False, u'nuclear': False, u'comprehendably': False, u'melrose': False, u'bruno': False, u'comprehendable': False, u'repetitively': False, u'nesmith': False, u'preminger': False, u'keynote': False, u'quirkyness': False, u'transpose': False, u'regretful': False, u'ifans': False, u'refuting': False, u'poof': False, u'lawsuit': False, u'staring': False, u'marty': False, u'hammy': False, u'commenting': False, u'swann': False, u'marts': False, u'metered': False, u'flanery': False, u'doorway': False, u'unearthing': False, u'vadar': False, u'conclude': False, u'roughed': False, u'ambushes': False, u'stylistically': False, u'confronts': False, u'600': False, u'blink': False, u'novalee': False, u'mailman': False, u'midland': False, u'cameraderie': False, u'catholicism': False, u'kahl': False, u'kahn': False, u'teamwork': False, u'eviscerated': False, u'yoram': False, u'feather': False, u'batinkoff': False, u'>': False, u'butchers': False, u'marcellus': False, u'altruist': False, u'sheepish': False, u'saxon': False, u'commuter': False, u'commutes': False, u'swarmed': False, u'coherence': False, u'taghmaoui': False, u'hateful': False, u'swindling': False, u'banish': False, u'miscommunicated': False, u'lecherous': False, u'reminiscence': False, u'gamesmanship': False, u'eaton': False, u'farsical': False, u'hahaha': False, u'pictures': False, u'fielding': False, u'miyake': False, u'stuffing': False, u'lawerence': False, u'nom': False, u'chaotic': False, u'petrucelli': False, u'workplaces': False, u'lascivious': False, u'norad': False, u'incense': False, u'chastising': False, u'diddley': False, u'ransom': False, u'tattoo': False, u'ostentatious': False, u'moves': False, u'grinder': False, u'pauly': False, u'subtitling': False, u'paull': False, u'painstaking': False, u'gibbs': False, u'dietl': False, u'pickin': False, u'chronicling': False, u'paula': False, u'unimaginable': False, u'complemented': False, u'forgotten': False, u'unsympathetic': False, u'censoring': False, u'fraidy': False, u'introduce': False, u'takers': False, u'berardinelli': False, u'blessedly': False, u'identity': False, u'infact': False, u'ofa': False, u'off': True, u'shotgun': False, u'dissing': False, u'patterns': False, u'rink': False, u'adrenaline': False, u'oft': False, u'comparisons': False, u'audio': False, u'tactfully': False, u'quentin': False, u'braniff': False, u'newest': False, u'obscenity': False, u'dissatisfying': False, u'wealthy': False, u'souped': False, u'not': True, u'coalwood': False, u'clocks': False, u'diedre': False, u'unmitigatedly': False, u'wether': False, u'web': False, u'tong': False, u'wee': False, u'hauntingly': False, u'wei': False, u'wen': False, u'toyed': False, u'undulating': False, u'wes': True, u'toni': False, u'terminating': False, u'wet': False, u'practise': False, u'villagers': False, u'tics': False, u'pieh': False, u'pied': False, u'crud': False, u'stooped': False, u'falters': False, u'mutations': False, u'crux': False, u'cruz': False, u'zimbabwe': False, u'hallucinates': False, u'atrophy': False, u'piet': False, u'tick': False, u'pier': False, u'pies': False, u'debatable': False, u'forcing': False, u'emma': False, u'bulge': False, u'gus': False, u'ring': False, u'mistrustful': False, u'flickering': False, u'become': False, u'emmy': False, u'assasination': False, u'palladino': False, u'drop': False, u'nutsy': False, u'knifepoint': False, u'brainerd': False, u'underwent': False, u'basque': False, u'_pick_chucky_up_': False, u'immortal': False, u'petey': False, u'gymnastics': False, u'choosing': False, u'flush': False, u'hissing': False, u'humming': False, u'recognition': False, u'delaurentiis': False, u'hipsters': False, u'mementos': False, u'buehler': False, u'cannibalistic': False, u'passion': False, u'bureaucratic': False, u'copulation': False, u'octogenarian': False, u'biology': False, u'uhhhm': False, u'brokering': False, u'pressure': False, u'sockful': False, u'amy': False, u'infiltrating': False, u'imaginary': False, u'coldly': False, u'homemaker': False, u'iwai': False, u'lifestyle': False, u'langer': False, u'burroughs': False, u'outshines': False, u'pillow': False, u'blackness': False, u'cycles': False, u'successors': False, u'sadoski': False, u'documentary': False, u'swimming': False, u'promiss': False, u'letters': False, u'miscegenation': False, u'rochon': False, u'fraud': False, u'hang': False, u'compadre': False, u'mojorino': False, u'privates': False, u'terminated': False, u'entices': False, u'letter_': False, u'brides': False, u'pairing': False, u'peters': False, u'heresy': False, u'indoctrinated': False, u'rhythmic': False, u'yarns': False, u'moonstruck': False, u'progenitor': False, u'cherokee': False, u'contradictory': False, u'jerkish': False, u'bagger': False, u'counterparts': False, u'zaniness': False, u'unformal': False, u'places': False, u'bloodline': False, u'congresswoman': False, u'excitement': False, u'placed': False, u'mouseketeer': False, u'tosses': False, u'problem': True, u'unsupportive': False, u'yeas': False, u'nurses': False, u'sodomy': False, u'cored': False, u'ink': False, u'_lot_': False, u'lobotomise': False, u'walters': False, u'effected': False, u'compared': False, u'nonetheless': False, u'deadly': False, u'purproses': False, u'lately': False, u'kerrigans': False, u'compares': False, u'details': False, u'boon': False, u'behold': False, u'vulgarize': False, u'illusion': False, u'ponytail': False, u'rebelled': False, u'repeat': False, u'jawa': False, u'zhou': False, u'softy': False, u'treason': False, u'allotting': False, u'impregnating': False, u'tinier': False, u'trunchbull': False, u'laude': False, u'exposure': False, u'searches': False, u'ustinov': False, u'disatisfaction': False, u'mishears': False, u'torrid': False, u'compete': False, u'lestat': False, u'villainous': False, u'searched': False, u'gardens': False, u'homerian': False}
#now lets do it for all docs
featuresets = [(find_features(rev),category) for (rev,category) in documents]
#we can split the feature sets into training and teesting datasets using sklearn
from sklearn import  model_selection
#define a seed for reproducibility
seed = 1
#split  the data into training & testing datasets
training, testing =  model_selection.train_test_split(featuresets,test_size =0.25,random_state=seed)
print(len(training))
print(len(testing))
1500
500
#how we use sklearn algos in nltk
from nltk.classify.scikitlearn import  SklearnClassifier
from sklearn.svm import SVC
model = SklearnClassifier(SVC(kernel = 'linear'))
#train model on training data
model.train(training)
<SklearnClassifier(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False))>
#test on testing dataset!
accuracy = nltk.classify.accuracy(model,testing)
print('svc accuracy:{}'.format(accuracy))
svc accuracy:0.66
#very good! higher than in the original tutorial!! )))
