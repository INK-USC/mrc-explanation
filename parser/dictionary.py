STRING2PREDICATE = {
     '"s': ['$Possessive'],
     ',': ['$Separator', '$Punctuate'],
     '/': ['$Separator'],
     '<': ['$LessThan'],
     '<=': ['$AtMost'],
     '<O>': ['$ArgY'],
     '<S>': ['$ArgX'],
     '=': ['$Equals'],
     '==': ['$Equals'],
     '>': ['$MoreThan'],
     '>=': ['$AtLeast'],
     'Between': ['$Between'],
     'OBJ': ['$ArgY'],
     'OBJ-ORGANIZATION': ['$ArgY'],
     'SUBJ': ['$ArgX'],
     'SUBJ-ORGANIZATION': ['$ArgX'],
     'The': ['$The'],
     'There': ['$There'],
     'X': ['$ArgX'],
     'Y': ['$ArgY'],
     'a': ['$The'],
     'admist': ['$Between'],
     'after': ['$Right'],
     'agencies': ['$OrganizationNER'],
     'agency': ['$OrganizationNER'],
     'all': ['$All', '$Adv'],
     'all capitalized': ['$Upper'],
     'all caps': ['$Upper'],
     'and': ['$And'],
     'another': ['$AtLeastOne'],
     'any': ['$Any'],
     'apart': ['$Apart'],
     'appear': ['$Is'],
     'appears': ['$Is'],
     'are': ['$Is'],
     'are identified': ['$Is'],
     'arg': ['$Arg'],
     'argument': ['$Arg'],
     'as': ['$Is', '$And'],
     'ask': ['$Contains'],
     'asks': ['$Contains'],
     'at least': ['$AtLeast'],
     'at most': ['$AtMost'],
     'away': ['$Apart'],
     'be': ['$Is'],
     'because': ['$Because'],
     'before': ['$Left'],
     'behind': ['$Right'],
     'between': ['$Between'],
     'both': ['$Both'],
     'but': ['$And'],
     'by': ['$By'],
     'capital': ['$Capital'],
     'capitalized': ['$Capital'],
     'capitals': ['$Capital'],
     'character': ['$Char'],
     'characters': ['$Char'],
     'city': ['$GPENER'],
     'cities': ['$GPENER'],
     'come': ['$Is'],
     'comes': ['$Is'],
     'companies': ['$OrganizationNER'],
     'company': ['$OrganizationNER'],
     'connect': ['$Link'],
     'connects': ['$Link'],
     'contain': ['$Contains'],
     'containing': ['$Contains'],
     'contains': ['$Contains'],
     'correct': ['$True'],
     'count': ['$Count'],
     'country': ['$GPENER'],
     'countries': ['$GPENER'],
     'date': ['$DateNER'],
     'dates': ['$DateNER'],
     'different': ['$NotEquals'],
     'different than': ['$NotEquals'],
     'directly': ['$Direct'],
     'each other': ['$EachOther'],
     'eachother': ['$EachOther'],
     'enclosed': ['$Between'],
     'end with': ['$EndsWith'],
     'ending': ['$Last'],
     'ending with': ['$EndsWith'],
     'ends with': ['$EndsWith'],
     'entities': ['$ArgXListAnd'],
     'equal': ['$Equals'],
     'equals': ['$Equals'],
     'exactly': ['$Equals'],
     'exist': ['$Exists'],
     'exists': ['$Exists'],
     'false': ['$False'],
     'final': ['$Last'],
     'followed by': ['$Left'],
     'following': ['$Right'],
     'follows': ['$Right'],
     'greater than': ['$MoreThan'],
     'greater than or equal': ['$AtLeast'],
     'identical': ['$Equals'],
     'if': ['$Because'],
     'immediately': ['$Direct'],
     'in': ['$In'],
     'in between': ['$Between'],
     'in front of': ['$Left'],
     'in the middle of': ['$Between'],
     'include': ['$Contains'],
     'includes': ['$Contains'],
     'incorrect': ['$False'],
     'institution': ['$OrganizationNER'],
     'institutions': ['$OrganizationNER'],
     'is': ['$Is'],
     'is found': ['$Is'],
     'is identified': ['$Is'],
     'is in': ['$In'],
     'is placed': ['$Is'],
     'is located': ['$Is'],
     'is referring to': ['$Contains'],
     'is similar to': ['$Equals'],
     'is stated': ['$Is'],
     'is used': ['$Is'],
     'it': ['$Sentence'],
     'larger than': ['$MoreThan'],
     'last': ['$Last'],
     'left': ['$Left'],
     'length': ['$Count'],
     'less than': ['$LessThan'],
     'less than or equal': ['$AtMost'],
     'letter': ['$Char'],
     'letters': ['$Char'],
     'link': ['$Link'],
     'links': ['$Link'],
     'location': ['$LocationNER'],
     'locations': ['$LocationNER'],
     'lower': ['$Lower'],
     'lower case': ['$Lower'],
     'lowercase': ['$Lower'],
     'man': ['$PersonNER'],
     'means': ['$Equals'],
     'mentioned': ['$Contains'],
     'mentions': ['$Contains'],
     'more than': ['$MoreThan'],
     'n"t': ['$Not'],
     'name': ['$PersonNER'],
     'neither': ['$None'],
     'next': ['$Within'],
     'no': ['$None', '$Int'],
     'no larger than': ['$AtMost'],
     'no less than': ['$AtLeast'],
     'no more than': ['$AtMost'],
     'no smaller than': ['$AtLeast'],
     'none': ['$None'],
     'nor': ['$Or'],
     'not': ['$Not'],
     'not any': ['$None'],
     'number': ['$Count', '$NumberNER', '$ChunkNum'],
     'numbers': ['$NumberNER'],
     'object': ['$ArgY'],
     'occur': ['$Is'],
     'occurs': ['$Is'],
     'of': ['$Of'],
     'one of': ['$Any'],
     'or': ['$Or'],
     'organization': ['$OrganizationNER'],
     'organizations': ['$OrganizationNER'],
     'pair': ['$Tuple'],
     'people': ['$PersonNER'],
     'person': ['$PersonNER'],
     'phrase': ['$Word'],
     'phrases': ['$Word'],
     'place': ['$ChunkPrep', '$LocationNER'],
     # 'places': ['$LocationNER'],
     'political': ['$NorpNER'],
     'politician': ['$NorpNER'],
     'preceded by': ['$Right'],
     'precedes': ['$Left'],
     'preceding': ['$Left'],
     'probably': ['$Adv'],
     'referred': ['$Contains'],
     'refers to': ['$Equals'],
     'refer to': ['$Equals'],
     'refers': ['$Contains'],
     'religious': ['$NorpNER'],
     'right': ['$Direct', '$Right'],
     'said': ['$Is'],
     'same': ['$Equals'],
     'same as': ['$Equals'],
     'sandwich': ['$SandWich'],
     'sandwiched': ['$SandWich'],
     'sandwiches': ['$SandWich'],
     'says': ['$Contains'],
     'separate': ['$Separate'],
     'separates': ['$Separate'],
     'should be': ['$Is'],
     'since': ['$Conjunction'],
     'smaller than': ['$LessThan'],
     'so': ['$Conjunction'],
     'start with': ['$StartsWith'],
     'starting with': ['$StartsWith'],
     'starts with': ['$StartsWith'],
     'states': ['$Contains'],
     'string': ['$Word'],
     'subject': ['$ArgX'],
     'time': ['$TimeNER'],
     'term': ['$Word'],
     'terms': ['$Word'],
     'text': ['$Sentence', '$Context'],
     'the': ['$The'],
     'answer': ['$Answer'],
     'question': ['$Question'],
     'sentence': ['$Sentence'],
     'context': ['$Context'],
     'the number of': ['$Numberof'],
     'them': ['$ArgXListAnd', '$ArgXListAnd'],
     'there': ['$There'],
     'therefore': ['$Conjunction'],
     'they': ['$ArgXListAnd', '$ArgXListAnd'],
     'to the left of': ['$Left'],
     'to the right of': ['$Right'],
     'token': ['$Token', '$Word'],
     'tokens': ['$Word'],
     'true': ['$True'],
     'tuple': ['$Tuple'],
     'upper': ['$Upper'],
     'upper case': ['$Upper'],
     'uppercase': ['$Upper'],
     'which': ['$Which'],
     'within': ['$LessThan'],
     'women': ['$PersonNER'],
     'word': ['$Word'],
     'words': ['$Word'],
     'wrong': ['$False'],
     'x': ['$ArgX'],
     'y': ['$ArgY'],
     'z': ['$ArgZ'],
     'noun': ['$ChunkNoun'],
     'noun phrase': ['$ChunkNoun'],
     'noun phrases': ['$ChunkNoun'],
     'proper noun': ['$ChunkNoun'],
     'preposition': ['$ChunkPrep'],
     'prepositional phrase': ['$ChunkPrep'],
     'verb': ['$ChunkVerb'],
     'verb phrase': ['$ChunkVerb'],
     'adjective': ['$ChunkAdj'],
     'adverb': ['$ChunkAdv'],

     'percentage': ['$PercentNER'],
     'percentage value': ['$PercentNER'],
     'percent': ['$PercentNER'],
     'someone': ['$PersonNER'],
     'right of': ['$Right'],
     'left of': ['$Left'],
     'is contained': ['$Contains_Passive'],
     'are contained': ['$Contains_Passive'],
     'statement': ['$Context'],
     'mean the same thing': ['$Equal_Post'],
     'has the same meaning': ['$Equal_Post'],
     'have the same meaning': ['$Equal_Post'],
     'requires': ['$Contains'],
     'is sandwiched': ['$Is'],
     'time period': ['$TimeNER'],
     'time interval': ['$TimeNER'],
     'period': ['$TimeNER'],
     'time frame': ['$TimeNER'],
     'is before': ['$Lneft'],
     'may be': ['$Is'],
     'amount': ['$NumberNER'],
     'age': ['$NumberNER'],
     'value': ['$NumberNER'],
     'ordinal number': ['$OrdinalNER'],
     'an': ['$The'],
     'about money': ['$MoneyNER'],
     'money': ['$MoneyNER'],
     'will be': ['$Is'],
     'begins with': ['$StartsWith'],
     'previous sentence': ['$PrevSentence'],
     'here': ['$Adv'],
     'present': ['$Is'],
     'presents': ['$Is'],
     'is present': ['$Is'],
     'are present': ['$Is'],
     'year': ['$DateNER'],
     'year number': ['$DateNER'],
     'also': ['$Adv'],
     'comma': ['$Comma'],
     'just': ['$Direct'],
     'has': ['$Contains'],
     'period of time': ['$TimeNER'],
     'nationality': ['NorpNER'],
     'question mark': ['$QMark'],
     'are before': ['$Left'],
     'which indicates': ['$And'],
     'right from': ['$Right'],
     'left from': ['$Left'],
     'amount of money': ['$MoneyNER'],
     'is expected to be': ['$Is'],
}

WORD_NUMBERS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven']
WORD2NUMBER = {elem: str(i) for i, elem in enumerate(WORD_NUMBERS)}

RAW_LEXICON = ''':- S,NP,N,PP
        VP :: S\\NP
        Det :: NP/N
        Adj :: N/N
        arg => NP {None}
        $And => var\\.,var/.,var {\\x y.'@And'(x,y)}
        $Or => var\\.,var/.,var {\\x y.'@Or'(x,y)}
        $Not => (S\\NP)\\(S\\NP) {None}
        $Not => (S\\NP)/(S\\NP) {None}
        $All => NP/N {None}
        $All => NP {None}
        $All => NP/NP {None}
        $Any => NP/N {None}
        $None => N {None}
        $Is => (S\\NP)/NP {\\y x.'@Is'(x,y)}
        #$Is => (S\\NP)/(S\\NP) {\\y x.'@Is'(x,y)}
        $Is => (S\\NP)/PP {\\y x.'@Is'(x,y)}   # word 'a' occurs between <S> and <O>
        $Is => (S\\NP)\\PP {\\y x.'@Is'(x,y)}  # between <S> and <O> occurs word 'a'
        $Is => (S\\PP)\\NP {\\x y.'@Is'(x,y)}  # between <S> and <O> word 'a' occurs
        $Exists => S\\NP/PP {\\y x.'@Is'(x,y)}
        $Int => Adj {None} #There are no words between <S> and <O>
        $AtLeastOne => NP/N {None}
        
        $Conjunction => (S\S)/S {\\x y.'@And'(x,y)}
        $Punctuate => S/S {\\x. x}
        $Punctuate => S\\S {\\x. x}

        $LessThan => PP/PP/N {\\x y.'@LessThan'(y,x)} #There are less than 3 words between <S> and <O>   
        $AtMost => PP/PP/N {\\x y.'@AtMost'(y,x)} #There are at most 3 words between <S> and <O>
        $AtLeast => PP/PP/N {\\x y.'@AtLeast'(y,x)} #same as above
        $MoreThan => PP/PP/N {\\x y.'@MoreThan'(y,x)} #same as above

        $LessThan => PP/N {\\x.'@LessThan1'(y,x)} #number of words between X and Y is less than 7.
        $AtMost => PP/N {\\x.'@AtMost1'(y,x)} 
        $AtLeast => PP/N {\\x.'@AtLeast1'(y,x)}   #same as above
        $MoreThan => PP/N {\\x.'@MoreThan1'(y,x)} #same as above

        $In => PP/NP {\\x.'@In0'(x)} 
        $In => (NP\\NP)/NP {\\y x. '@In'(x, y)}
        # $In => (NP/NP)\\NP {\\x y. x}
        $Contains => S\\NP/NP {\\y x.'@Is'(y, '@In0'(x))} #The sentence contains two words
        # $Contains_Passive => S\\NP/NP {\\y x.'@Is'(x, '@In'(y))} #Y is contained in the sentence
        $Contains_Passive => S\\NP/PP {\\y x. '@Is'(x, y)}
        # $Contains => S\\NP/NP {\\y x.'@In'(y, x)} #The sentence contains two words
        $Separator => var\\.,var/.,var {\\x y.'@And'(x,y)} #connection between two words
        $EachOther => N {None}
        $Token => N {\\x.'@Word'(x)}
        $Word => NP/N {\\x.x}
        $Word => NP/NP {\\x.x}

        $Word => N {'tokens'} #There are no more than 3 words between <S> and <O>
        $Word => NP {'tokens'} #There are no more than 3 words between <S> and <O>

        $Char => N {None} #same as above
        $StartsWith => S\\NP/NP {\\y x.'@StartsWith'(x,y)}
        $EndsWith => S\\NP/NP {\\y x.'@EndsWith'(x,y)}
        $Left => PP/NP {\\x.'@Left'(x)} # the word 'a' is before <S>
        $Left => (S\\NP)/NP {\\y x.'@Left0'(y,x)}  #Precedes
        $Left => NP/NP {\\x.'@Left1'(x)}
        $Right => PP/NP {\\x.'@Right'(x)}# the word 'a' ia after <S>
        $Right => (S\\NP)/NP {\\y x.'@Right0'(y,x)} 
        $Right => NP/NP {\\x.'@Right1'(x)}
        #$Within => ((S\\NP)\\(S\\NP))/NP {None} # the word 'a' is within 2 words after <S>
        #$Within => (NP\\NP)/NP {None}
        $Within => PP/PP/N {\\x y.'@AtMost'(y,x)} #Does Within has other meaning.
        $Sentence => NP {'Sentence'}

        $Between => (S/S)/NP {\\x y.'@Between'(x,y)}
        $Between => S/NP {\\x.'@Between'(x)}
        $Between => PP/NP {\\x.'@Between'(x)}
        $Between => (NP\\NP)/NP {\\x y.'@Between'(x,y)}

        $PersonNER => NP {'@NER'('PERSON')}
        $GPENER => NP {'@NER'('GPE')}
        $LocationNER => NP {'@NER'('LOCATION')}
        $DateNER => NP {'@NER'('DATE')}
        $NumberNER => NP {'@NER'('NUMBER')}
        $OrdinalNER => NP {'@NER'('ORDINAL')}
        $TimeNER => NP {'@NER'('TIME')}
        $OrganizationNER => NP {'@NER'('ORGANIZATION')}
        $PercentNER => NP {'@NER'('PERCENT')}
        $MoneyNER => NP {'@NER'('MONEY')}
        
        $ChunkNoun => NP {'@Chunk'('N')}
        $ChunkVerb => NP {'@Chunk'('V')}
        $ChunkPrep => NP {'@Chunk'('P')}
        $ChunkAdj => NP {'@Chunk'('ADJ')}
        $ChunkAdv => NP {'@Chunk'('ADV')}
        $ChunkNum => NP {'@Chunk'('NUM')}
        
        $ArgX => NP {'X'}
        $ArgY => NP {'Y'}
        $ArgZ => NP {'Z'}
        $ArgX => NP {'@In'('X','all')}
        $ArgY => NP {'@In'('Y','all')}
        $ArgZ => NP {'@In'('Z','all')}

        $that => NP/N {None}
        $Apart => (S/PP)\\NP {None}
        $Direct => PP/PP {\\x.'@Direct'(x)} # the word 'a' is right before <S>   
        $Direct => (S\\NP)/PP {\\y x.'@Is'(x,'@Direct'(y))}
        $Last => Adj {None}
        $There => NP {'There'}
        $By => S\\NP\\PP/NP {\\z f x.'@By'(x,f,z)} #precedes sth by 10 chatacters       
        $By => (S\\NP)\\PP/(PP/PP) {\\F x y.'@Is'(y,F(x))} #precedes sth by no more than10 chatacters        
        $By => PP\\PP/(PP/PP) {\\F x. F(x)} #occurs before by no...

        $Numberof => NP/PP/NP {\\x F.'@NumberOf'(x,F)}

        # $Of => PP/NP {\\x.'@Range0'(x)} # the word 'x' is at most 3 words of Y     
        $Of => NP/NP {\\x.x} #these two are designed to solve problems like $Is $Left $Of and $Is $Left
        $Of => N/N {\\x.x}
        $Char => NP/N {None}
        $ArgX => N {'ArgX'}
        $ArgY => N {'ArgY'}
        $Answer => NP {'Answer'}
        $Question => NP {'Question'}
        $Sentence => NP {'Sentence'}
        $Context => NP {'Context'}
        $PrevSentence => NP {'PrevSentence'}
        $Link => (S\\NP)/NP {\\x y.'@Is'(y,'@Between'(x))}
        $SandWich => (S\\NP)/NP {\\x y.'@Is'(x,'@Sandwich'(y))}
        $The => N/N {\\x.x}
        $The => NP/NP {\\x.x}
        
        $Equals => (S/NP)\\NP {\\x y. '@Equal'(x,y)}
        $Adv => NP/NP {\\x.x}
        $Adv => NP\\NP {\\x.x}
        $Adv => S/S {\\x.x}
        $Adv => S\\S {\\x.x}
        $Separate => (S/NP)\\NP {\\y x. '@Is'(x,'@Between'(y))}
        $Both => PP/PP {\\x.x}
        $Equal_Post => S\\NP {\\x.'@Equal0'(x)}
        $In => (S\\S)/NP {\\y x. '@In1'(x, y)}
        $In => (PP/NP)/S {\\x y. '@In1'(x, y)}
        $Comma => NP {'[COMMA]'}
        $QMark => NP {'?'}
        '''

QUESTION_WORDS = [
    "(what|what's)",
    "(who|who's)",
    "where",
    "when",
    "which",
    "whose",
    "whom",
    "how",
    "why",
    "(can|could|may|might|should)",
    "(is|are|were|was)",
    "(will|would)",
    "(do|does|did)",
    "(has|have|had)",
    "(name|identify|describe|define)"
]