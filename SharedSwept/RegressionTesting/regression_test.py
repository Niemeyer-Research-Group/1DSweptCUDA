import json
import numpy as np
import os
import sys

def test_eval(a,b):
    a = np.array(a)
    b = np.array(b)
    rt = .001
    at = .0001
    num = np.sum(np.abs(a-b))
    if np.any(np.isnan(a)):
        print "Failed!  There's a nan."
        return num
    if not a.size == b.size:
        print "Failed!  They're not the same size"
        return num
    if np.any(np.logical_not(np.isclose(a,b,rtol=rt,atol=at))):
        truth = np.logical_not(np.isclose(a,b,rtol=rt,atol=at))
        idx = np.where(truth)
        tols = at + rt*abs(b[truth])
        diff = abs(a[truth]-b[truth])
        print "Failed!  The results are not the same at {0} places out of {1}".format(np.sum(truth), a.size)
        # if np.sum(truth)>9:
        #     print "Measured", a[truth][0:-1:np.sum(truth)/10]
        #     print "Saved", b[truth][0:-1:np.sum(truth)/10]
        #     print "Difference", diff[0:-1:np.sum(truth)/10]
        #     print "Tolerance", tols[0:-1:np.sum(truth)/10]
        #     print "Index", idx[0:-1:np.sum(truth)/10]
        # else:
        #     print "Measured", a[truth]
        #     print "Saved", b[truth]
        #     print "Difference", diff
        #     print "Tolerance", tols
        #     print "Index", idx

        # I = raw_input("Enter 1 to stop: ")
        # if int(I):
        #     sys.exit(1)


        return num

    print "It passed! "
    return num



def consistency_test(prob,div,tend,data):
    sourcepath = os.path.abspath(os.path.dirname(__file__))
    jsonfile = os.path.join(sourcepath,"testData.json")
    sdiv = str(div)
    stend = str(tend)
    rd = dict()
    n = -1
    if not os.path.isfile(jsonfile):
        rd[prob] = dict()
        rd[prob][sdiv] = dict()
        rd[prob][sdiv][stend] = data
        regd = open(jsonfile,"a+")
        json.dump(rd, regd)
        regd.close()
        print "File created"
        return n

    regd = open(jsonfile,"r")
    rd = json.load(regd)
    #Now test if the run has a dictionary analog.
    if prob in rd.keys():
        if sdiv in rd[prob].keys():
            if stend in rd[prob][sdiv].keys():
                n = test_eval(data,rd[prob][sdiv][stend])
                return n
            else:
                rd[prob][sdiv][stend] = data
        else:
            rd[prob][sdiv] = dict()
            rd[prob][sdiv][stend] = data
    else:
        rd[prob] = dict()
        rd[prob][sdiv] = dict()
        rd[prob][sdiv][stend] = data

    regd.close()

    regd = open(jsonfile,"w+")

    json.dump(rd, regd)
    print "This dataset has been added to regression test."
    regd.close()
    return n
