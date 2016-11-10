from dl_convergence import *

from joblib import Parallel, delayed
import multiprocessing

if __name__ == "__main__":
    verbosity=0
    max_steps = 2000
    eval_steps = 100      
    th = 0.5
    eta = 0.1 # 0.1 * m/s
    data = []
    batch_size = 256 # \Om(m * s)
    #f = open('am_dls.txt', 'w')
    f = None
    def func(m,s):
        n = m/2 
        q = s/m
        #printv((s,m,n,q),verbosity,1)
        A = make_A(m,n,verbosity)
        d = []
        for (init,st) in [(init_close(A, 0), "0"),
                          (init_close(A, 0.05), "0.05"),
                          (init_close(A, 0.1), "0.1"),
                          (init_close(A, 0.2), "0.2"),
                          (init_close(A, 0.5), "0.5"),
                          (1/math.sqrt(n)*np.random.randn(m,n).astype(np.float32), "random"), 
                          (1/math.sqrt(n)*np.random.randn(2*m,n).astype(np.float32), "overrandom"), 
                          (make_data_set_pm1(A,m,m,s), "samples"),
                          (make_data_set_pm1(A,2*m,m,s), "oversamples")]:
            loss = np.nan
            eta=0.1
            printv(st, verbosity, 1)
            #if f!=None:
            #    f.write(st)
            #    f.write("\n")
            while np.isnan(loss):
                (loss, B, Bn, mins1, mins2, mins3, AB) = \
                    train_dl_and_eval(A, m, s, batch_size, max_steps, eval_steps, eta, f, init, st, 
                                      th=0.5, verbosity=verbosity)
                eta = eta/10
            eta=0.1
            (nB, nloss) = train_nn_dl(A, np.transpose(init), m, n,s, batch_size, max_steps, eta, th, eval_steps)
            (nloss, nB, nBn, nmins1, nmins2, nmins3, nAB) = evals(A, nB, nloss, f, verbosity)
            #eta = eta/10
            d.append(((m,n,s, st, loss, mins1, mins2, mins3, A, B, AB),(m,n,s,st,nloss,nmins1, nmins2, nmins3, A, nB, nAB)))
        return d
    num_cores = multiprocessing.cpu_count()
    li = [(50*2**k, 2**k2) for k in range(5) for k2 in range(int(math.ceil(math.log(50*2**k,2)/2))+1)]
    datas = concat(Parallel(n_jobs = num_cores)(delayed(func)(t[0],t[1]) for t in li))
    (data1, data2) = unzip(datas)
    with open('am_dls1.pickle', 'wb') as f:
        pickle.dump(data1, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    with open('nn_dls1.pickle', 'wb') as f2:
        pickle.dump(data2, f2, pickle.HIGHEST_PROTOCOL)
    f2.close()
    #f.close()
    #with open('am_dls_data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    #    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
