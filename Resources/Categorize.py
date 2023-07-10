import re
import unidecode
import unicodedata

def getSymbols(string):
    syms = re.compile(r'(S[0-9a-fA-F]{3}[0-5]{1}[0-9a-fA-F]{1}[0-9]{3}x[0-9]{3})|([BLMR][0-9]{3}x[0-9]{3})|(S[0-9a-fA-F]{3}[0-5]{1}[0-9a-fA-F]{1})')
    s = syms.findall(string)
    r = []
    for item in s:
        if (item[0] != ''):
            r.append(item[0])
    return r

def clean_text(string):
    string = string.lower()
    string = unidecode.unidecode(string)
    string = re.sub(r"[^a-zA-Z0-9.']", ' ', string)

    return string

def N0Symbols(string):
    s = string[1:]
    if(int(s, 16) >= int('100', 16) and int(s, 16) <= int('204', 16)):
        return 'hand'
    if(int(s, 16) >= int('205', 16) and int(s, 16) <= int('2f6', 16)):
        return 'movement'
    if(int(s, 16) >= int('2f7', 16) and int(s, 16) <= int('2fe', 16)):
        return 'dynamic'
    if(int(s, 16) >= int('2ff', 16) and int(s, 16) <= int('36c', 16)):
        return 'head'
    if(int(s, 16) >= int('36d', 16) and int(s, 16) <= int('37e', 16)):
        return 'body'
    if(int(s, 16) >= int('37f', 16) and int(s, 16) <= int('386', 16)):
        return 'location'
    if(int(s, 16) >= int('387', 16) and int(s, 16) <= int('38b', 16)):
        return 'punctuation'

def N1Hands(string):
    s = string[1:]
    if(int(s, 16) >= int('100', 16) and int(s, 16) <= int('204', 16)):
        if(int(s, 16) >= int('100', 16) and int(s, 16) < int('10E', 16)):
            return 0
        if(int(s, 16) >= int('10E', 16) and int(s, 16) < int('11E', 16)):
            return 1
        if(int(s, 16) >= int('11E', 16) and int(s, 16) < int('144', 16)):
            return 2
        if(int(s, 16) >= int('144', 16) and int(s, 16) < int('14C', 16)):
            return 3
        if(int(s, 16) >= int('14C', 16) and int(s, 16) < int('186', 16)):
            return 4
        if(int(s, 16) >= int('186', 16) and int(s, 16) < int('1A4', 16)):
            return 5
        if(int(s, 16) >= int('1A4', 16) and int(s, 16) < int('1BA', 16)):
            return 6
        if(int(s, 16) >= int('1BA', 16) and int(s, 16) < int('1CD', 16)):
            return 7
        if(int(s, 16) >= int('1CD', 16) and int(s, 16) < int('1F5', 16)):
            return 8
        if(int(s, 16) >= int('1F5', 16) and int(s, 16) < int('205', 16)):
            return 9
    else:
        None

def LeftRight(string):
    s = string
    if(int(s, 16)<= int('7', 16)):
        return 'right'
    else:
        return 'left'


def N1Movements(string):
    s = string[1:]
    if(int(s, 16) >= int('205', 16) and int(s, 16) <= int('2f6', 16)):
        if(int(s, 16) >= int('205', 16) and int(s, 16) < int('216', 16)):
            return 0
        if(int(s, 16) >= int('216', 16) and int(s, 16) < int('22A', 16)):
            return 1
        if(int(s, 16) >= int('22A', 16) and int(s, 16) < int('255', 16)):
            return 2
        if(int(s, 16) >= int('255', 16) and int(s, 16) < int('265', 16)):
            return 3
        if(int(s, 16) >= int('265', 16) and int(s, 16) < int('288', 16)):
            return 4
        if(int(s, 16) >= int('288', 16) and int(s, 16) < int('2A6', 16)):
            return 5
        if(int(s, 16) >= int('2A6', 16) and int(s, 16) < int('2B7', 16)):
            return 6
        if(int(s, 16) >= int('2B7', 16) and int(s, 16) < int('2D5', 16)):
            return 7
        if(int(s, 16) >= int('2D5', 16) and int(s, 16) < int('2E3', 16)):
            return 8
        if(int(s, 16) >= int('2E3', 16) and int(s, 16) < int('2f7', 16)):
            return 9
    else:
        None

def N1Dynamic(string):
    s = string[1:]
    if(int(s, 16) >= int('2f7', 16) and int(s, 16) <= int('2fe', 16)):
        return 0
    else:
        None

def N1Head(string):
    s = string[1:]
    if(int(s, 16) >= int('2ff', 16) and int(s, 16) <= int('36c', 16)):
        if(int(s, 16) >= int('2FF', 16) and int(s, 16) < int('30A', 16)):
            return 0
        if(int(s, 16) >= int('30A', 16) and int(s, 16) < int('32A', 16)):
            return 1
        if(int(s, 16) >= int('32A', 16) and int(s, 16) < int('33B', 16)):
            return 2
        if(int(s, 16) >= int('33B', 16) and int(s, 16) < int('359', 16)):
            return 3
        if(int(s, 16) >= int('359', 16) and int(s, 16) < int('36D', 16)):
            return 4
    else:
        None

def N1Body(string):
    s = string[1:]
    if(int(s, 16) >= int('36d', 16) and int(s, 16) <= int('37e', 16)):
        if(int(s, 16) >= int('36D', 16) and int(s, 16) < int('376', 16)):
            return 0
        if(int(s, 16) >= int('376', 16) and int(s, 16) < int('37f', 16)):
            return 1
    else:
        None

def N1Location(string):
    s = string[1:]
    if(int(s, 16) >= int('37f', 16) and int(s, 16) <= int('386', 16)):
        return 0
    else:
        None

def N1Punctuation(string):
    s = string[1:]
    if(int(s, 16) >= int('387', 16) and int(s, 16) <= int('38b', 16)):
        return 0
    else:
        None

def eachCat(type, string):
    if(type == 'hand'):
        return N1Hands(string)
    if(type == 'movement'):
        return N1Movements(string)
    if(type == 'dynamic'):
        return N1Dynamic(string)
    if(type == 'head'):
        return N1Head(string)
    if(type == 'body'):
        return N1Body(string)
    if(type == 'location'):
        return N1Location(string)
    if(type == 'punctuation'):
        return N1Punctuation(string)
    return None

def orientacao(cod):
    if(cod == 0):
        return 'PalmFrontWall'
    if(cod == 1):
        return 'SideFrontWall'
    if(cod == 2):
        return 'BackFrontWall'
    if(cod == 3):
        return 'PalmToptFloor'
    if(cod == 4):
        return 'SideTopFloor'
    if(cod == 5):
        return 'BackTopFloor'

    return None

def details(cod):
    symbols = getSymbols(cod)
    s=[]
    for i in symbols:
        type = N0Symbols(i[0:4])
        aux = {}
        aux = {
            'x': i[6:9],
            'y': i[10:13],
            'full': i[0:6],
            'base': i[0:4],
            'N0':N0Symbols(i[0:4]),
            'N1':eachCat(type, i[0:4]),
            'N2':i[4],
            'CM': i[0:4] if N0Symbols(i[0:4]) == 'hand' else None,
            'rotation': i[5],
            'left_right': LeftRight(i[5]) if N0Symbols(i[0:4]) == 'hand' else None,
            'orientacao': orientacao(i[4]),
        }
        s.append(aux)
    return s