from matplotlib.ticker import PercentFormatter

def generate_probs(n_steps, model):

    probs = {}

    for i in range(n_steps+1):

        probs[i] = {}

        for j in range(-i,i+1):

            probs[i][j] = model(i,j)

    return probs


def build_tree_and_yield_torch(probs, n_steps, R0, delta_r, delta_t, torch):

    times = []
    yields = []

    for Tn in range(1,n_steps+1):

        B = {Tn:{j:torch.tensor(1.0) for j in probs[Tn]}}

        for i in range(Tn-1,-1,-1):

            B[i] = {}

            for j in probs[i]:

                Pu,Pm,Pd = probs[i][j]

                R = R0 + j*delta_r

                val = (
                    Pu*B[i+1].get(j+1,torch.tensor(0.0))
                    +Pm*B[i+1].get(j,torch.tensor(0.0))
                    +Pd*B[i+1].get(j-1,torch.tensor(0.0))
                )

                B[i][j] = torch.exp(-R*delta_t)*val

        P0 = B[0][0]

        t = Tn*delta_t

        y = -torch.log(P0)/t

        times.append(t)
        yields.append(y)

    return torch.tensor(times),torch.stack(yields)


def build_idi_tree(step, IDI0, R0, delta_r, delta_t, torch, initial_idi_value=None):

    if initial_idi_value is None:
        initial_idi_value = IDI0

    tree = [{"j": 0, "IDI": torch.tensor(initial_idi_value)}]
    levels = [tree]

    for i in range(step):

        new_nodes = []

        for node in tree:

            j = node["j"]
            idi = node["IDI"]

            r = R0 + j * delta_r
            idi_next = idi * (1 + r * delta_t)

            new_nodes.append({"j": j + 1, "IDI": idi_next})
            new_nodes.append({"j": j,     "IDI": idi_next})
            new_nodes.append({"j": j - 1, "IDI": idi_next})

        levels.append(new_nodes)
        tree = new_nodes

    return levels


def build_atm_strikes(n_steps, delta_t, target_yield, IDI0, torch):

    strikes = []

    for Tn in range(1,n_steps+1):

        t = Tn*delta_t

        y = target_yield[Tn-1]

        K = IDI0*(1+y*t)

        strikes.append(K)

    return torch.stack(strikes)


def price_option_step(probs, step, K, IDI0, R0, delta_r, delta_t, torch):

    tree = build_idi_tree(step, IDI0, R0, delta_r, delta_t, torch)

    V = []

    for p in tree[-1]:
        payoff = torch.maximum(p["IDI"] - K, torch.tensor(0.0))
        V.append(payoff)

    V = torch.stack(V)

    for i in reversed(range(step)):

        newV = []

        for k, p in enumerate(tree[i]):

            j = p["j"]
            r = R0 + j*delta_r

            Pu, Pm, Pd = probs[i][j]

            child = k*3

            Vu = V[child]
            Vm = V[child+1]
            Vd = V[child+2]

            val = (Pu*Vu + Pm*Vm + Pd*Vd)/(1+r*delta_t)
            newV.append(val)

        V = torch.stack(newV)

    return V[0]


def price_all_options_2(probs, strikes, IDI0, R0, delta_r, delta_t, torch):

    prices = []

    for i in range(len(strikes)):
        step = i + 1
        K = strikes[i]

        price = price_option_step(probs, step, K, IDI0, R0, delta_r, delta_t, torch)
        prices.append(price)

    return torch.stack(prices)


def plot_loss(loss_history, plt):

    plt.figure()
    plt.plot(loss_history)
    plt.title("Perda de Treinamento")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.grid()
    plt.show()


def plot_option_prices(C_model, C_market, np, plt):

    steps = np.arange(1, len(C_market) + 1)

    plt.figure(figsize=(8,5))
    ax = plt.gca()

    model_color = '#1f77b4'
    market_color = 'black'

    ax.plot(steps, C_model.detach().numpy(),
            'o', color=model_color, label="Árvore Trinomial Neural", zorder=2)

    ax.plot(steps, C_market.detach().numpy(),
            'x', color=market_color, label="Mercado", zorder=3)

    ax.set_title("Preços de Opções sobre IDI")
    ax.set_xlabel("Passo de Tempo (i)")
    ax.set_ylabel("Preço")

    ax.legend()
    ax.grid(True)

    plt.show()


def plot_yield_curve(yield_model, target_yield, np, plt, y_lim=None):

    steps = np.arange(1,len(target_yield)+1)

    plt.figure()

    plt.plot(steps, yield_model.detach(), label="Árvore Trinomial Neural", marker="o")
    plt.plot(steps, target_yield.detach(), 'x--', label="Mercado")
    plt.title("Ajuste da ETTJ")

    plt.xlabel("Passo de Tempo (i)")
    plt.ylabel("Taxa de Juros")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))

    if y_lim is not None:
        plt.ylim(y_lim)

    plt.legend()
    plt.grid()
    plt.show()


def plot_trinomial_tree2(probs, n_steps, r0, delta_r, np, plt):

    import matplotlib.colors as mcolors

    tree = []
    tree.append({0: 1.0})

    for j in range(1, n_steps + 1):
        prev_layer = tree[-1]
        curr_layer = {}

        increments = np.array([+1, 0, -1])

        for m_prev, p_prev in prev_layer.items():

            if (j - 1) in probs and m_prev in probs[j - 1]:
                transition_probs = probs[j - 1][m_prev].detach().numpy()
            else:
                transition_probs = np.array([0.0, 0.0, 0.0])

            for idx, inc in enumerate(increments):
                m_new = m_prev + inc
                p_new = p_prev * transition_probs[idx]
                curr_layer[m_new] = curr_layer.get(m_new, 0) + p_new

        tree.append(curr_layer)

    plt.figure(figsize=(12, 8))

    cmap = plt.cm.Reds
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for j in range(1, n_steps + 1):

        increments = np.array([+1, 0, -1])
        prev_nodes = sorted(tree[j-1].keys())

        for m_prev in prev_nodes:

            r_prev = r0 + m_prev * delta_r

            if (j - 1) in probs and m_prev in probs[j - 1]:
                node_probs = probs[j - 1][m_prev].detach().numpy()
            else:
                node_probs = np.array([0.0, 0.0, 0.0])

            for idx, inc in enumerate(increments):

                p = node_probs[idx]
                if p <= 0:
                    continue

                m_new = m_prev + inc
                r_new = r0 + m_new * delta_r

                color = cmap(norm(p))
                linewidth = 1.5 + 2 * p

                plt.plot([j-1, j], [r_prev, r_new],
                         color=color,
                         linewidth=linewidth,
                         alpha=0.9,
                         zorder=1)

                if j == 1:
                    y_offset = delta_r * 0.25 * inc

                    plt.text(j - 0.5,
                             (r_prev + r_new) / 2 + y_offset,
                             f'{p:.2f}',
                             fontsize=8,
                             ha='center',
                             va='center',
                             color='black',
                             bbox=dict(facecolor='white',
                                       alpha=0.8,
                                       edgecolor='none'),
                             zorder=2)

    for j in range(n_steps + 1):
        xs = [j] * len(tree[j])
        ys = [r0 + m * delta_r for m in sorted(tree[j].keys())]

        plt.scatter(xs, ys,
                    s=90,
                    color='blue',
                    edgecolor='black',
                    zorder=3)

    plt.title("Árvore Trinomial Neural")
    plt.xlabel("Passo de Tempo (i)")
    plt.ylabel("Taxa de Juros R(i,j)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    plt.grid(True)
    plt.xticks(range(n_steps + 1))

    plt.show()


def plot_option_prices_surface(surface_model, surface_market, strike_surface, plt):

    plt.figure(figsize=(8,5))
    ax = plt.gca()

    steps_list = sorted(list(surface_model.keys()))

    model_color = 'black'
    market_color = '#1f77b4'

    maturity_centers = []
    maturity_labels = []

    for idx, step in enumerate(steps_list):

        Ks = strike_surface[step].detach().numpy()
        C_model = surface_model[step].detach().numpy()
        C_market = surface_market[step].detach().numpy()

        if idx == 0:
            ax.plot(Ks, C_model, 'x--', color=model_color, label="Árvore Trinomial Neural")
            ax.plot(Ks, C_market, 'o-', color=market_color, label="Mercado")
        else:
            ax.plot(Ks, C_model, 'x--', color=model_color)
            ax.plot(Ks, C_market, 'o-', color=market_color)

        maturity_centers.append(Ks.mean())
        maturity_labels.append(f"T{step}")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Preço da Opção")
    ax.set_title("Preços de Opções por Vencimento")

    ax.legend()
    ax.grid(True)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(maturity_centers)
    ax_top.set_xticklabels(maturity_labels)
    ax_top.set_xlabel("Maturidade")

    plt.show()


def price_surface(probs, strike_surface, torch, IDI0, R0, delta_r, delta_t):

    surface = {}

    for step in strike_surface:

        Ks = strike_surface[step]

        prices = []

        for K in Ks:

            p = price_option_step(probs, step, K, IDI0, R0, delta_r, delta_t, torch)
            prices.append(p)

        surface[step] = torch.stack(prices)

    return surface


def surface_loss(surface, market_surface, torch):

    loss = torch.tensor(0.0)

    for step in surface:

        loss += torch.mean((surface[step]-market_surface[step])**2)

    return loss/len(surface)


def price_all_options_surface(probs, n_steps, strike_surface, torch, IDI0, R0, delta_r, delta_t):

    surface_model = {}

    for step in range(1,n_steps+1):

        strikes = strike_surface[step]

        prices = []

        for k in strikes:

            price = price_option_step(probs, step, k.item(), IDI0, R0, delta_r, delta_t, torch)
            prices.append(price)

        surface_model[step] = torch.stack(prices)

    return surface_model