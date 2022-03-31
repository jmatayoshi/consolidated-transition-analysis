import numpy as np
import scipy.stats
import statsmodels.api as sm
from joblib import Parallel, delayed, wrap_non_picklable_objects

@wrap_non_picklable_objects
def generate_conditional_sequence(seq_length, states, inv_states,
                                  cond_base_rates, base_rates=[]):
    if len(base_rates) == 0:
        base_rates = np.ones(len(states)) / len(states)
    # Sample first state
    seq = [states[np.random.multinomial(1, base_rates).argmax()]] 
    for i in range(seq_length - 1):
        # Get index of previous state
        prev_ind = inv_states[seq[-1]]
        # Sample next state based on conditional base rates
        # np.random.multinomial is faster here than np.random.choice
        seq.append(states[
            np.random.multinomial(1, cond_base_rates[prev_ind]).argmax()])
    return seq

@wrap_non_picklable_objects
def generate_sequence(seq_length, states, base_rates):
    seq = list(np.random.choice(states,
                                size=seq_length,
                                p=base_rates)
    )
    return seq

@wrap_non_picklable_objects
def generate_dependent_sequence(seq_length, states, inv_states,
                                base_rates, dependent_rates):
    # np.random.multinomial is faster than np.random.choice
    state_ind = np.arange(len(states))
    new_rates = dependent_rates.copy()    
    for i in range(len(states)):
        new_rates[i, :] += base_rates
        new_rates[i, :] /= new_rates[i, :].sum()
    
    seq = [states[np.argmax(np.random.multinomial(1, base_rates))]] 

    for i in range(seq_length - 1):        
        temp_rates = new_rates[inv_states[seq[-1]], :]
        seq.append(states[np.argmax(np.random.multinomial(
            1, temp_rates))])
    return seq

@wrap_non_picklable_objects
def compute_cond_probs(seq, states):
    prev_count = {a: 0 for a in states}
    cond_count = {a: {b: 0 for b in states} for a in states}
    for i in np.arange(1, len(seq)):
        for a in states:
            if seq[i - 1] == a:
                prev_count[a] += 1
                for b in states:
                    if seq[i] == b:
                        cond_count[a][b] += 1
                        break
                break   
    res = []
    for a in states:
        for b in states:
            if prev_count[a] > 0:
                res.append(cond_count[a][b] / prev_count[a])
            else:
                res.append(np.nan)
    return np.array(res).reshape((len(states), len(states)))

@wrap_non_picklable_objects
def compute_L(seq, A, B):
    A_prev = 0
    B_next_A_prev = 0
    B_next = 0

    for i in np.arange(1, len(seq)):
        if seq[i] == B:
            B_next += 1
        if seq[i - 1] == A:
            A_prev += 1
            if seq[i] == B:
                B_next_A_prev += 1

    P_B_next = B_next / (len(seq) - 1)

    if P_B_next == 1:
        return np.nan
    elif A_prev == 0:
        return np.nan
    else:
        P_B_next_A_prev = B_next_A_prev / A_prev
        return (P_B_next_A_prev - P_B_next) / (1 - P_B_next)
   

def fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable(),
                       family=sm.families.Binomial()):
    """ Fit marginal model to sequential data.

    Parameters
    ----------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    cov : statsmodels class, optional
        One of the following working dependence structures for GEE model: 
            sm.cov_struct.Independence()
            sm.cov_struct.Exchangeable()
            sm.cov_struct.Autoregressive()
    cov : statsmodels class, optional
        One of the following exponential family link functions for GEE model: 
            sm.families.Binomial()
            sm.families.Gamma()
            sm.families.Gaussian()
            sm.families.InverseGaussian()
            sm.families.NegativeBinomial()
            sm.families.Poisson()
            sm.families.Tweedie()


    Returns
    -------
    beta_coef : float
         Coefficient from the single independent variable in the GEE model
    p_val : float
         The p-value returned from a two-tailed t-test on beta_coef
    P_A_B : float
         Estimated probability of transitioning to A given that the starting 
         state is B
    P_A_not_B : float
         Estimated probability of transitioning to A given that the starting 
         state is not B
    """
    md = sm.GEE(
        y, X, groups,
        cov_struct=cov,
        family=family
    )    
    fit_res = md.fit()
    beta_coef = fit_res.params[1]
    p_val = fit_res.pvalues[1]  
    P_A_B = md.predict(fit_res.params, exog=np.array([1, 1]))
    P_A_not_B = md.predict(fit_res.params, exog=np.array([1, 0]))

    return beta_coef, p_val, P_A_B, P_A_not_B, fit_res.conf_int()[1]

def df_to_y_X(df, a, b, min_length=0, excluded_states=[]):
    """ Function for turning a DataFrame of sequential data into the appropriate
    format for GEE model

    Parameters
    ----------
    df : DataFrame
        First column contains a student index, second column an affect state
        Rows are grouped based on the student index and ordered sequentially 
        Example:
            1,CON
            1,FLO
            1,FRU
            2,FLO
            2,FLO
            3,BOR
    a : str/float
        Starting state
    b : str/float
        Ending state
    min_length : int, optional
        Sequences less than min_length are excluded

    Returns
    -------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    """
    y = []
    X = []
    seq_ind = []
    all_transitions = True
    if len(excluded_states) > 0:
        all_transitions = False    
    for i in np.unique(df.iloc[:, 0].values):
        pos = np.flatnonzero(df.iloc[:, 0].values == i)
        if len(pos) >= min_length:
            for j in range(len(pos) - 1):
                if (
                        df.iloc[pos[j], 0] == df.iloc[pos[j + 1], 0] and
                        (
                            all_transitions or
                            df.iloc[pos[j + 1]] not in excluded_states
                        )
                ):
                    seq_ind.append(df.iloc[pos[j], 0])
                    if df.iloc[pos[j], 1] == a:
                        X.append([1, 1])
                    else:
                        X.append([1, 0])            
                    if df.iloc[pos[j + 1], 1] == b:
                        y.append(1)
                    else:
                        y.append(0)                
    return np.array(y), np.array(X), np.array(seq_ind)

@wrap_non_picklable_objects
def sequences_to_y_X(seq_list, a, b, min_length=0, excluded_states=[]):
    """ Function for turning a list of sequences into the appropriate
    format for GEE model

    Parameters
    ----------
    seq_list : list of lists
        Each entry in the list is a sequence (list) of transition states
        Example:
            [
                ['A', 'C', 'C', 'B', 'C'],
                ['B', 'C', 'A', 'C'],
                ['C', 'C', 'C', 'B', 'B', 'A']
            ]
    a : str/float
        Starting state
    b : str/float
        Ending state
    min_length : int, optional
        Sequences less than min_length are excluded

    Returns
    -------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    """
    y = []
    X = []
    seq_ind = []
    all_transitions = True
    if len(excluded_states) > 0:
        all_transitions = False
    for i in range(len(seq_list)):
        curr_seq = seq_list[i]
        if len(curr_seq) >= min_length:
            for j in range(len(curr_seq) - 1):
                if all_transitions or curr_seq[j + 1] not in excluded_states:                            
                    seq_ind.append(i)                
                    if curr_seq[j] == a:
                        X.append([1, 1])
                    else:
                        X.append([1, 0])
                    if curr_seq[j + 1] == b:
                        y.append(1)
                    else:
                        y.append(0)

    return np.array(y), np.array(X), np.array(seq_ind)

def run_simulations(
        num_trials=10000,
        base_rates=np.array([0.5, 0.5]),
        seq_lengths=np.arange(3, 151)):
    """ Run numerical experiments 
    
    Experiment 1 parameters (results shown in Figures 1 and 2): 
        num_trials=10000,
        base_rates=np.array([0.5, 0.5]),
        seq_lengths=np.arange(3, 151)

    Experiment 2 parameters (results shown in Figure 3): 
        num_trials=10000,
        base_rates=np.array([0.6, 0.2, 0.1, 0.1]),
        seq_lengths=np.arange(3, 151)
    
    Returns
    -------
    Average conditional probabilities for A-->A and A-->B
    GEE estimated conditional probabilities for A-->A and A-->B
    L values for A-->A and A-->B
    GEE \beta_1 values for A-->A and A-->B
    
    """    
    L_AA = []
    L_AB = []
    P_AA = []
    P_AB = []
    gee_beta_AA = []
    gee_beta_AB = []
    gee_P_AA = []
    gee_P_AB = []
    
    states = np.arange(base_rates.shape[0])

    for seq_len in seq_lengths:
        if seq_len % 5 == 0:
            print('Current sequence length = ' + str(seq_len))

        seq_list = []
        for i in range(num_trials):
            seq_list.append(generate_sequence(seq_len, states, base_rates))
        curr_AA = []
        curr_AB = []
        curr_P_AA = []
        curr_P_AB = []        
        for i in range(len(seq_list)):
            curr_seq = seq_list[i]
            curr_AA.append(compute_L(curr_seq, states[0], states[0]))
            curr_AB.append(compute_L(curr_seq, states[0], states[1]))
            res = compute_cond_probs(curr_seq, states)
            curr_P_AA.append(res[0, 0])
            curr_P_AB.append(res[0, 1])            
        L_AA.append(np.nanmean(curr_AA))
        L_AB.append(np.nanmean(curr_AB))
        P_AA.append(np.nanmean(curr_P_AA))
        P_AB.append(np.nanmean(curr_P_AB))            

        y, X, groups = sequences_to_y_X(seq_list, states[0], states[0])                
        res = fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable())
        gee_beta_AA.append(res[0])
        gee_P_AA.append(res[2])
        
        y, X, groups = sequences_to_y_X(seq_list, states[0], states[1])
        res = fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable())
        gee_beta_AB.append(res[0])
        gee_P_AB.append(res[2])
        
    return [
        # conditional probabilities
        P_AA, P_AB,
        # GEE estimated conditional probabilities
        gee_P_AA, gee_P_AB,
        # L statistic values
        L_AA, L_AB,
        # GEE \beta_1 coefficients
        gee_beta_AA, gee_beta_AB,
    ]

@wrap_non_picklable_objects
def get_counts(seq, states):
    next_count = {a: 0 for a in states}
    cond_count = {a: {b: 0 for b in states} for a in states}
    num_tr = len(seq) - 1
    # Compute next and conditional counts
    for i in np.arange(1, len(seq)):
        for a in states:
            if seq[i - 1] == a:
                for b in states:
                    if seq[i] == b:
                        cond_count[a][b] += 1
                        next_count[b] += 1
                        break
                break
    cond_count_list = []
    for a in states:
        for b in states:
            cond_count_list.append(cond_count[a][b])

    next_count_list = []
    for a in states:
        next_count_list.append(next_count[a])
    return next_count_list, cond_count_list

@wrap_non_picklable_objects
def compile_sequence_counts(seq_list, states):
    next_counts = []
    cond_counts = []    
    for seq in seq_list:
        count_res = get_counts(seq, states)
        next_counts.append(count_res[0])
        cond_counts.append(count_res[1])
    return np.array(next_counts), np.array(cond_counts)

@wrap_non_picklable_objects
def get_L_star_vals(a, b, next_counts, cond_counts, use_mean_rates=True):
    num_states = next_counts.shape[1]
    # Column indices where next != a (i.e., transitions in T_{A_complement})
    a_comp_ind = (
        np.array([i for i in range(num_states) if i != a])
    )
    # Count transitions where prev == a and next != a
    a_comp_cond_sum = cond_counts[:, a_comp_ind + a*num_states].sum(axis=1)
    if use_mean_rates:      
        # Compute L_star using base rates averaged over the whole sample
        # of sequences; note that as opposed to the computation of
        # L_star below, we only exclude samples with P(b|a) == nan; that is,
        # we only exclude sequences with no transitions from a to another state
        sample_pos = np.flatnonzero(
            a_comp_cond_sum > 0
        )
        # Compute mean base rate of b restricted to transitions with next != a
        modified_mean_base_rate = np.mean(
            next_counts[sample_pos, b] /
            next_counts[sample_pos, :][:, a_comp_ind].sum(axis=1)
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )
        L_star_vals = (
            (cond_rates - modified_mean_base_rate)
            / (1 - modified_mean_base_rate)
        )
    else:
        # Compute L_star using base rates from each individual sequence

        # Column indices where next != a and next != b
        a_b_comp_ind = (
            np.array([i for i in range(num_states) if i != a and i != b])
        )
        # Count transitions where next != a or next != b
        a_b_comp_sum = next_counts[:, a_b_comp_ind].sum(axis=1)
        # Count transitions where next != a        
        a_comp_sum = next_counts[:, b] + a_b_comp_sum
        # Find samples where:
        #  (a) P(b|a) != nan
        #  (b) P(b) < 1
        sample_pos = np.flatnonzero(
            (a_comp_cond_sum > 0) & (a_b_comp_sum > 0)            
        )        
        # Compute base rates of b restricted to transitions with next != a
        modified_base = (
            next_counts[sample_pos, b] / a_comp_sum[sample_pos]
        )
        # Compute conditional rate of b restricted to transitions with next != a
        cond_rates = (
            cond_counts[sample_pos, a*num_states + b] /
            a_comp_cond_sum[sample_pos]
        )       
        L_star_vals = (
            (cond_rates - modified_base)
            / (1 - modified_base)
        )
    return L_star_vals

def run_sequence_sims(rate=0.0, seq_length=20, num_trials=50,
                      num_runs=10000, vary_cluster=False,
                      verbose=5, n_jobs=1, multiple_shifts=False):

    states = ['A', 'B', 'C', 'D', 'E']
    base_rates = np.ones(len(states)) / len(states)
    num_states = len(states)
    inv_states = {}
    for i in range(num_states):
        inv_states[states[i]] = i
        
    dep_rates = np.zeros((num_states, num_states))
    # When rate > 0 we have four false null hypotheses
    dep_rates[0, 1] = rate
    dep_rates[0, 3] = -1*rate
    dep_rates[2, 1] = -1*rate
    dep_rates[2, 3] = rate

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(sequence_sim)(
        seq_length,
        num_trials,
        states,
        inv_states,
        base_rates,
        dep_rates,
        num_states,
        vary_cluster=vary_cluster,
        multiple_shifts=multiple_shifts)
        for x in range(num_runs))
    return out

@wrap_non_picklable_objects
def sequence_sim(seq_length, num_trials, states, inv_states, base_rates,
                 dep_rates, num_states, shift_list=[0.04, 0.08, 0.12],
                 vary_cluster=False, multiple_shifts=False):
    sim_count = 0
    while True:
        # Simulation will restart if one of the models throws an error
        sim_count += 1
        try:
            pval_gee = []
            se_gee = []
            beta_gee = []
            pval_no_self = []
            se_no_self = []
            beta_no_self = []    
            pval_L = []
            test_L = []
            val_L = []  
            num_groups = []
            num_groups_no_self = []
            pval_lr = []
            beta_lr = []
            seq_list = []
            for j in range(num_trials):
                shift = shift_list[np.random.randint(len(shift_list))]
                shifted_rates = base_rates.copy()
                shift_ind = np.random.permutation(len(states))        
                shifted_rates[shift_ind[0]] += shift
                shifted_rates[shift_ind[1]] -= shift
                if multiple_shifts:
                    shift = shift_list[np.random.randint(len(shift_list))]        
                    shifted_rates[shift_ind[2]] += shift
                    shifted_rates[shift_ind[3]] -= shift
                cluster_shift = 0
                cluster_shift_list = np.arange(-35, 36)
                if vary_cluster:
                    cluster_shift = cluster_shift_list[np.random.randint(
                        len(cluster_shift_list))]
                seq_list.append(generate_dependent_sequence(seq_length+cluster_shift,
                                                            states, inv_states,
                                                            shifted_rates, dep_rates))
            next_counts, cond_counts = compile_sequence_counts(seq_list, states)
            for m in range(num_states):
                for n in range(num_states):
                    if m != n:
                        # Compute L_star values
                        res = get_L_star_vals(m, n, next_counts, cond_counts,
                                              use_mean_rates=False)
                        t_res = scipy.stats.ttest_1samp(res, 0)

                        pval_L.append(t_res[1])
                        test_L.append(t_res[0])
                        val_L.append(np.mean(res))

                        # Run marginal model, excluding self-transitions
                        y, X, groups = sequences_to_y_X(seq_list, states[m], states[n],
                                                        excluded_states=[states[m]])
                        num_groups_no_self.append(len(np.unique(groups)))
                        md = sm.GEE(
                            y, X, groups,
                            cov_struct=sm.cov_struct.Exchangeable(),
                            family=sm.families.Binomial(),
                        )                
                        fit_res = md.fit(maxiter=60, cov_type='bias_reduced')

                        pval_no_self.append(fit_res.pvalues[1])
                        beta_no_self.append(fit_res.params[1])
                        se_no_self.append([np.sqrt(fit_res.cov_robust_bc[1, 1]),
                                                  np.sqrt(fit_res.cov_robust[1, 1])])

                    # Run marginal model, including self-transitions
                    y, X, groups = sequences_to_y_X(seq_list, states[m], states[n])
                    num_groups.append(len(np.unique(groups)))
                    md = sm.GEE(
                        y, X, groups,
                        cov_struct=sm.cov_struct.Exchangeable(),
                        family=sm.families.Binomial()
                    )                
                    fit_res = md.fit(maxiter=60, cov_type='bias_reduced')

                    pval_gee.append(fit_res.pvalues[1])
                    beta_gee.append(fit_res.params[1])
                    se_gee.append([np.sqrt(fit_res.cov_robust_bc[1, 1]),
                                         np.sqrt(fit_res.cov_robust[1, 1])])
            # Break loop once all models have successfully been fit
            break
        except np.linalg.LinAlgError:
            pass
    return (
        # GEE p-values, standard errors, and beta coefficients        
        np.array(pval_gee), np.array(se_gee), np.array(beta_gee),
        # L_star p-values, test statistics, and computed values        
        np.array(pval_L), np.array(test_L), np.array(val_L),
        # GEE p-values, standard errors, and beta coefficients        
        np.array(pval_no_self), np.array(se_no_self), np.array(beta_no_self),
        # Simulation attempts, sequences, no-self transition sequences
        sim_count, num_groups, num_groups_no_self,
    )

def compute_md(beta, se, num_groups, ci=False):
    # Compute p-values and 95% confidence intervals using the correction
    # outlined by Mancel and DeRouen (2001)
    res = np.array([
        2*(1 - scipy.stats.t.cdf(np.abs(beta[i] / se[i]),
                                 num_groups[i] - 2)) for i in range(len(beta))
    ])
    if ci:
        t = scipy.stats.t.ppf(0.975, num_groups - 2)
        ci_res = np.c_[beta - se*t, beta + se*t]
        return res, ci_res
    else:
        return res

def compute_df(beta, se, num_groups):
    # Compute p-values using the degrees-of-freedom correction proposed by
    # MacKinnon and White (1985)
    res = np.array([
        2*(1 - scipy.stats.t.cdf(np.abs(beta[i] /
            (np.sqrt(num_groups[i] / (num_groups[i] - 2))*se[i])),
                                 num_groups[i] - 2))
        for i in range(len(beta))           
    ])
    return res
            
def analyze_sequence_results(sim_data, dependence=0, L_star=False,
                               thresh_list = [0.05, 0.1, 0.15]):
    '''
    Analyze the output from run_sequence_sims.  Returns the estimated FDR 
    values from the BH and BY procedures, the TPR values if dependence == True,
    and in all cases the error bounds for the 99% confidence intervals.
    '''
    model_list = ['robust', 'MR', 'DF', 'L*', 'robust', 'MR', 'DF']    
    num_rows = len(thresh_list)
    if dependence:
        false_array = np.zeros((len(model_list), 4*len(thresh_list)))
    res_array = np.zeros((len(model_list), 4*len(thresh_list)))
    
    self_num_pairs = 5*5
    self_true_null = np.array([0, 2, 4,
                               5, 6, 7, 8, 9,
                               10, 12, 14,
                               15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24])
    no_self_num_pairs = 5*4
    no_self_true_null = np.array([1, 3,
                                  4, 5, 6, 7,
                                  8, 11,
                                  12, 13, 14, 15,
                                  16, 17, 18, 19])
   
    res_str = ''

    for m_ind in range(len(model_list)):
        if m_ind == 0:
            res = (1 - scipy.stats.norm.cdf(np.abs(
                [sim_data[i][2] / sim_data[i][1][:, 1] for i in range(len(sim_data))]
            )))*2
        elif m_ind == 1:
            res = compute_md(
                np.array([sim_data[i][2] for i in range(len(sim_data))]),
                np.array([sim_data[i][1][:, 0] for i in range(len(sim_data))]),
                np.array([sim_data[i][10] for i in range(len(sim_data))])                
            )
        elif m_ind == 2:
            res = compute_df(
                np.array([sim_data[i][2] for i in range(len(sim_data))]),
                np.array([sim_data[i][1][:, 1] for i in range(len(sim_data))]),
                np.array([sim_data[i][10] for i in range(len(sim_data))])                
            )            
        elif m_ind == 3:
            res = np.array([sim_data[i][3] for i in range(len(sim_data))])
        elif m_ind == 4:
            res = (1 - scipy.stats.norm.cdf(np.abs(
                [sim_data[i][8] / sim_data[i][7][:, 1]
                 for i in range(len(sim_data))])))*2
        elif m_ind == 5:
            res = compute_md(
                np.array([sim_data[i][8] for i in range(len(sim_data))]),
                np.array([sim_data[i][7][:, 0] for i in range(len(sim_data))]),
                np.array([sim_data[i][11] for i in range(len(sim_data))])                
            )            
        elif m_ind == 6:
            res = compute_df(
                np.array([sim_data[i][8] for i in range(len(sim_data))]),
                np.array([sim_data[i][7][:, 1] for i in range(len(sim_data))]),
                np.array([sim_data[i][11] for i in range(len(sim_data))])                
            )                        
        if m_ind < 3:
            self_transitions = True
            true_null = self_true_null
            num_pairs = self_num_pairs
        else:
            self_transitions = True
            true_null = no_self_true_null
            num_pairs = no_self_num_pairs
        if dependence:
            null_ind = true_null
            false_null = np.setdiff1d(np.arange(num_pairs), null_ind)
        else:
            null_ind = np.arange(num_pairs)
            false_null = np.array([])
        res_str += '\n' + model_list[m_ind] + '\n'
        false_str = ''
        
        for thresh_ind in range(len(thresh_list)):
            thresh = thresh_list[thresh_ind]
            Q_array = np.zeros((len(sim_data), 2))
            pos_array = np.zeros((len(sim_data), 2))
            for i in range(len(sim_data)):            
                curr_res = sm.stats.multipletests(res[i],
                                                  method='fdr_bh',
                                                  alpha=thresh)        
                if len(np.flatnonzero(curr_res[0])) > 0:
                    Q_array[i, 0] = (
                        len(np.flatnonzero(curr_res[0][null_ind]))/
                        len(np.flatnonzero(curr_res[0]))
                    )
                if dependence:
                    pos_array[i, 0] = (
                        len(np.flatnonzero(curr_res[0][false_null]))/
                        len(false_null)
                    )                
                curr_res = sm.stats.multipletests(res[i],
                                                  method='fdr_by',
                                                  alpha=thresh)
                if len(np.flatnonzero(curr_res[0])) > 0:
                    Q_array[i, 1] = (
                        len(np.flatnonzero(curr_res[0][null_ind]))/
                        len(np.flatnonzero(curr_res[0]))
                    )
                if dependence:
                    pos_array[i, 1] = (
                        len(np.flatnonzero(curr_res[0][false_null]))/
                        len(false_null)
                    )
            # Estimated FDR for BH (fdr[0]) and BY (fdr[1])
            fdr = np.mean(Q_array, axis=0)
            se = scipy.stats.sem(Q_array, axis=0)
            # 99% error for BH               
            err0 = se[0]*scipy.stats.t.ppf((1+0.99)/2, 
                                         Q_array.shape[0] - 1)
            # 99% error for BY                
            err1 = se[1]*scipy.stats.t.ppf((1+0.99)/2, 
                                         Q_array.shape[0] - 1)
            if thresh_ind > 0:
                res_str += ' '
            res_str += '{}: {:.3f} +/- {:.3f},  {:.3f} +/- {:.3f}'.format(
                thresh, fdr[0], err0, fdr[1], err1
            )        
            res_array[m_ind, thresh_ind*4:(thresh_ind + 1)*4] = np.array(
                [fdr[0], err0, fdr[1], err1])
            if dependence:
                # Estimated TPR for BH (tpr[0]) and BY (tpr[1])
                tpr = np.mean(pos_array, axis=0)
                se = scipy.stats.sem(pos_array, axis=0)
                # 99% error for BH               
                err0 = se[0]*scipy.stats.t.ppf((1+0.99)/2, 
                                             pos_array.shape[0] - 1)
                # 99% error for BY                
                err1 = se[1]*scipy.stats.t.ppf((1+0.99)/2, 
                                             pos_array.shape[0] - 1)
                if thresh_ind > 0:
                    false_str += ' '
                false_str += '{}: {:.3f} +/- {:.3f},  {:.3f} +/- {:.3f}'.format(
                    thresh, tpr[0], err0, tpr[1], err1
                )
                false_array[m_ind, thresh_ind*4:(thresh_ind + 1)*4] = np.array(
                    [tpr[0], err0, tpr[1], err1])                
        res_str += '\n'
        if dependence:
            res_str += false_str + '\n'
    print(res_str)
    if dependence:
        return res_array, false_array
    else:
        return res_array

def run_no_self_simulations(
        num_trials=10000,
        base_rates=np.array([0.25, 0.25, 0.25, 0.25]),        
        seq_lengths=np.arange(3, 151)):
    """ Run numerical experiments with self-transitions removed (Section 3.2)
    
    Three states:
        num_trials=10000,
        base_rates=np.ones(3) / 3,
        seq_lengths=np.arange(3, 151)

    Four states:
        num_trials=10000,
        base_rates=np.ones(4) / 4,
        seq_lengths=np.arange(3, 151)
    
    Returns
    -------
    GEE estimated conditional probabilities for A-->B
    GEE \beta_1 values and confidence intervals for A-->B
    L* values for A-->B    
    """    
    L_star_AB = []
    gee_beta_AB = []
    gee_P_AB = []
    gee_conf_AB = []
    
    states = np.arange(base_rates.shape[0])
    num_states = len(states)
        
    for seq_len in seq_lengths:
        if seq_len % 5 == 0:
            print('Current sequence length = ' + str(seq_len))

        seq_list = []
        for i in range(num_trials):
            seq_list.append(generate_sequence(seq_len, states, base_rates))

        next_counts, cond_counts = compile_sequence_counts(seq_list, states)
        
        # Compute L_star values
        L_star_res = get_L_star_vals(0, 1, next_counts, cond_counts,
                                     use_mean_rates=False)
        L_star_AB.append(np.nanmean(L_star_res))

    
        y, X, groups = sequences_to_y_X(seq_list, states[0], states[1],
                                        excluded_states=[states[0]])
        gee_res = fit_marginal_model(y, X, groups,
                                     cov=sm.cov_struct.Exchangeable())
        gee_beta_AB.append(gee_res[0])
        gee_P_AB.append(gee_res[2])
        gee_conf_AB.append(gee_res[4])
       

    return [
        # GEE estimated conditional probabilities, \beta_1 values, and CIs
        gee_P_AB, gee_beta_AB, gee_conf_AB,
        # L* values
        L_star_AB
    ]
