from model import Generator
from model import Discriminator
import torch
import torch.nn.functional as F
import numpy as np
import copy
import os


LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def encrypt_subs(initial):
    initial = initial.lower()
    output = ""

    key = 'qwertyuiopasdfghjklzxcvbnm'  # key for encrypt

    shift = []

    for j in range(len(key)):
        x = ord(key[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1

    return output


def decrypt(initial):
    initial = initial.lower()
    output = ""

    key = 'kxvmcnophqrszyijadlegwbuft'  # inverse key for decrypt

    shift = []

    for j in range(len(key)):
        x = ord(key[j]) - 97
        shift.append(x)

    cnt = 0
    for char in initial:
        if char in LETTERS:
            output += LETTERS[shift[LETTERS.index(char)]]
            cnt += 1

    return output



def getkeymatrix(key, key_len):
    key_mat = [[0]*key_len for i in range(key_len)]

    key = key.lower()

    for i in range(key_len):
        for j in range(key_len):
            key_mat[i][j] = ord(key[key_len*i+j]) - 97

    inv = get_inv(key_mat)

    return key_mat


def get_inv(key_mat):

    a = key_mat[0][0]
    d = key_mat[1][1]
    b = key_mat[0][1]
    c = key_mat[1][0]

    tmp = a*d - b*c

    ret = 0

    if (gcd(tmp, 26) != 1):
        print("plz fix keys")
    else:
        for i in range(26):
            if ((tmp * i) % 26 == 1):
                ret = i

    return ret


def gcd(a, b):
    return b if a == 0 else gcd(b % a, a)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # the normalize code -> t.sub_(m).div_(s)
        return tensor


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, data_loader_test, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader
        self.data_loader_test = data_loader_test

        # Model configurations.
        self.c_dim = config.c_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Miscellaneous.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Build the model and tensorboard.
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim)
        self.D = Discriminator(
            self.d_conv_dim, self.c_dim
        )
        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )

        self.G.to(self.device, dtype=torch.float)
        self.D.to(self.device, dtype=torch.float)

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device, dtype=torch.float))
        return c_trg_list

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device, dtype=torch.float)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
            allow_unused=False
        )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        # return torch.mean((dydx_l2norm - 1) ** 2)
        return torch.mean((1-dydx_l2norm) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def StoE2(self, x):
        """from Simplex to Embedding"""
        x_total = torch.reshape(x, (x.size(0), 26, 100))
        emb = self.G.main[0].embed
        concat = self.G.main[0].concat
        #emb = self.G.main.module[0].embed

        xs = torch.empty(x.size(0), 256, 100)

        for q in range(x.size(0)):  # for each batch
            xs[q] = torch.matmul(emb, x_total[q])  # 256 * 26 * 26 * 100

        concat_batch = torch.empty(x.size(0), 256, 100)
        concat_batch = concat_batch.to(self.device)
        for i in range(x.size(0)):
            concat_batch[i] = concat

        xs = xs.to(self.device, dtype=torch.float)

        xs = torch.cat((xs, concat_batch), dim=1)

        return xs   # (batch, 256, 100)

    def Simplex(self, x):   # x : (batch_size, 1, sample * vocab_size)
        """Initialize to Simplex"""
        """continuous relaxation"""
        ############ one-hot simplex for prof partition ############
        arr = [0.,     0.0392, 0.0784, 0.1176, 0.1569, 0.1961,
               0.2353, 0.2745, 0.3137, 0.3529, 0.3922, 0.4314,
               0.4706, 0.5098, 0.5490, 0.5882, 0.6275, 0.6667,
               0.7059, 0.7451, 0.7843, 0.8235, 0.8627, 0.9020,
               0.9412, 1.0000]

        #x_total = []

        x_total = torch.empty(0, dtype=torch.float)
        for q in range(x.size(0)):  # x.size(0) = batchsize
            #simplex = []
            simplex = torch.empty(0, dtype=torch.float)
            for i in range(x.size(2)):
                tmp = [0 for k in range(len(arr))]
                for j in range(len(arr)):
                    if round(float(x[q][0][i]), 4) == round(float(arr[j]), 4):
                        tmp[j] = 1
                        # simplex.append(tmp)
                        tmp = torch.from_numpy(np.array(tmp, dtype=np.float32))
                        simplex = torch.cat((simplex, tmp))
                        break

            simplex = torch.from_numpy(np.array(simplex, dtype=np.float32))
            x_total = torch.cat((x_total, simplex))

        x_total = torch.reshape(x_total, (x.size(0), 100, 26))
        x_total = torch.transpose(x_total, 1, 2)
        x_total = x_total.to(self.device, dtype=torch.float)

        return x_total  # (batch, 26, 100)

    def myonehot(self, x):
        """onehot vector for x which is thde output from self.G"""
        onehot = torch.zeros_like(x)

        for q in range(onehot.size(0)):
            tt = torch.argmax(x[q], dim=0)
            for e in range(onehot.size(2)):
                onehot[q][tt[e]][e] = 1.
        onehot = onehot.to(self.device, dtype=torch.float)

        return onehot

    def train(self):
        """Train model within a single dataset."""
        # Set data loader.
        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Start training from scratch or resume training.
        start_iters = 0

        accu_tmp = []   # for Caeser
        accu_tmp2 = []  # for Vig
        accu_tmp3 = []  # for Hill

        # accu test from PT to CT
        accu_tmp01 = []  # Caeser
        accu_tmp02 = []  # Vig
        accu_tmp03 = []  # Subs

        # accu test for each cipher emulation
        accu_CtoV = []
        accu_CtoS = []
        accu_VtoC = []
        accu_VtoS = []
        accu_StoC = []
        accu_StoV = []

        # Start training.
        print("Start training...")
        # start_time = time.time()
        for i in range(start_iters, self.num_iters, 10000):

            '''warmup_constant in lr_schemes.py'''

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Compute loss with real images.

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = torch.reshape(
                x_real, (x_real.size(0), x_real.size(1), 100))

            # x_real = x_real.to(self.device)  # Input images.
            # Original domain labels.
            c_org = c_org.to(self.device, dtype=torch.float)
            # Target domain labels.
            c_trg = c_trg.to(self.device, dtype=torch.float)
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # USING Simplex function
            ######################################
            x_groundtruth = self.Simplex(x_real)  # batch,  26, 100
            x_real_tmp = self.StoE2(x_groundtruth)  # batch, 256, 100

            # embedding line : (100, 26) * (26, 256) = (100, 256)
            out_src, out_cls = self.D(x_real_tmp)

            d_loss_real = torch.mean((out_src-1)**2)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            x_groundtruth = x_groundtruth.to(self.device)
            x_real_tmp = x_real_tmp.to(self.device)

            # Compute loss with fake images.
            x_fake = self.G(x_groundtruth, c_trg)
            x_fake = self.StoE2(x_fake)

            out_src, out_cls = self.D(x_fake.detach())

            d_loss_fake = torch.mean((out_src)**2)

            # Compute loss for gradient penalty.
            # modified from (x_real.size(0), 1, 1, 1) to (x_real.size(0), 1, 1)
            alpha = torch.rand(x_groundtruth.size(0), 1, 1).to(
                self.device, dtype=torch.float)
            x_hat = (alpha * x_real_tmp.data + (1 - alpha) * x_fake.data).requires_grad_(
                True
            )

            out_src, _ = self.D(x_hat)

            out_src = torch.reshape(
                out_src, (out_src.size(0), out_src.size(2)))

            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = (
                0.5 * (d_loss_real			# E[D(x)] (should be minus)
                       + d_loss_fake)			# E[D(G, (x, c))] (should be plus)
                + self.lambda_cls * d_loss_cls  # L^r_cls
                + self.lambda_gp * d_loss_gp  # Wasserstein penalty
            )

            self.d_optimizer.zero_grad()

            d_loss.backward(retain_graph=True)
            self.d_optimizer.step()

            loss_d = d_loss.item()

            # print(i)

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            x_cipher_fake = self.G(x_groundtruth, c_trg)
            x_cipher_rfake = self.StoE2(x_cipher_fake)
            out_src, out_cls = self.D(x_cipher_rfake)
            g1_loss_fake = torch.mean((out_src-1)**2)
            g_loss_cls = self.classification_loss(out_cls, label_trg)

            x_cipher_fake = self.myonehot(x_cipher_fake)

            x_reconst = self.G(x_cipher_fake, c_org)
            x_reconst = self.myonehot(x_reconst)

            g_loss_rec = torch.mean(torch.abs(x_cipher_fake - x_reconst))

            g_loss = (
                g1_loss_fake
                + self.lambda_rec * g_loss_rec
                + self.lambda_cls * g_loss_cls
            )

            self.g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.g_optimizer.step()

            loss_g = g_loss.item()

            print("Epoch", i, "D loss:", loss_d, "G_loss:", loss_g)

            with torch.no_grad():

                #### for testline ##############
                data_loader_test = self.data_loader_test

                data_iter_test = iter(data_loader_test)
                x_fixed_test, c_org_test = next(data_iter_test)

                iden = 0
                while (iden == 0):

                    id0 = []
                    for idxstest0 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest0] == 0.):
                            id0.append(idxstest0)

                    id1 = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == 1.):
                            id1.append(idxstest)
                        continue

                    id2 = []
                    for idxstest2 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest2] == 2.):
                            id2.append(idxstest2)
                        continue

                    id3 = []
                    for idxstest3 in range(c_org_test.size(0)):
                        if (c_org_test[idxstest3] == 3.):
                            id3.append(idxstest3)
                        continue

                    if len(id1) == 0 or len(id2) == 0 or len(id3) == 0:
                        iden == 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = iter(data_loader_test)
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                x_fixed_test = torch.reshape(
                    x_fixed_test, (x_fixed_test.size(0), x_fixed_test.size(1), 100))

            c_fixed_list_test = self.create_labels(c_org_test, self.c_dim)

            # using simplex fct.
            x_fixed_total_test = self.Simplex(x_fixed_test)

            ###########################################################################
            ############################# cipher to plain #############################
            ###########################################################################
            x_fixed_fake_test = self.G(
                x_fixed_total_test, c_fixed_list_test[0])
            x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

            e = 0
            accu = 0
            while e < len(id1):
                list4 = ''  # for idx(Caeser)
                list5 = ''  # for recovered PT from idx(Caeser)

                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                            list4 += (chr(97+w))
                        if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                            list5 += (chr(97+w))

                # Decrypt line
                last = ''   # for recovered Caeser

                for q in range(len(list4)):
                    tmp = (ord(list4[q]) - 97 - 3) % 26
                    tmp = chr(tmp + 97)
                    last += tmp

                cnt = 0
                for q in range(len(list4)):
                    if (last[q] != list5[q]):
                        cnt += 1
                    else:
                        continue
                accu += float((len(list4) - cnt) / len(list4))
                e += 1
            accu = accu / len(id1)
            accu_tmp.append(accu)

            e = 0
            accu2 = 0
            while e < len(id2):
                list6 = ''  # for idx2(Vigenere)
                list7 = ''  # for recovered PT from idx2(Vigenere)

                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                            list6 += (chr(97+w))
                        if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                            list7 += (chr(97+w))

                # Decrypt line
                last2 = ''   # for recovered Vigenere

                for q in range(len(list6)):
                    if q % 4 == 0:
                        tmp2 = (ord(list6[q]) - 97 - 3) % 26
                        tmp2 = chr(tmp2 + 97)
                        last2 += tmp2
                    elif q % 4 == 1:
                        tmp2 = (ord(list6[q]) - 97 - 4) % 26
                        tmp2 = chr(tmp2 + 97)
                        last2 += tmp2
                    elif q % 4 == 2:
                        tmp2 = (ord(list6[q]) - 97 - 5) % 26
                        tmp2 = chr(tmp2 + 97)
                        last2 += tmp2
                    else:
                        tmp2 = (ord(list6[q]) - 97 - 6) % 26
                        tmp2 = chr(tmp2 + 97)
                        last2 += tmp2

                cnt = 0
                for q in range(len(list6)):
                    if (last2[q] != list7[q]):
                        cnt += 1
                    else:
                        continue
                accu2 += float((len(list6) - cnt) / len(list6))
                e += 1
            accu2 = accu2 / len(id2)
            accu_tmp2.append(accu2)

            e = 0
            accu3 = 0
            while e < len(id3):
                list8 = ''  # for Substitution
                list9 = ''  # for recovered PT from list8

                for q in range(100):
                    for w in range(26):
                        if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                            list8 += (chr(97+w))
                        if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                            list9 += (chr(97+w))

                # Decrypt line
                last3 = decrypt(list8)    # for recovered subs

                cnt = 0
                for q in range(len(list8)):
                    if (last3[q] != list9[q]):
                        cnt += 1
                    else:
                        continue
                accu3 += float((len(list7) - cnt) / len(list8))
                e += 1
            accu3 = accu3 / len(id3)
            accu_tmp3.append(accu3)

            print("Ceaser to plain : ", accu)
            print("Vigenere to plain : ", accu2)
            print("Subs to plain : ", accu3)

            ###########################################################################
            ############################# plain to cipher #############################
            ###########################################################################

            # Target domain is (tt+1) (1 : Caeser, 2 : Vigenere, 3 : Subs)
            for tt in range(3):
                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt+1])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt+1)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                # print(len(ids))

                e = 0
                accu1 = 0  # -> Caeser
                accu2 = 0  # -> Vigenere
                accu3 = 0  # -> Substitution
                while e < len(id0):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id0[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id0[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Caeser

                    if (tt+1) == 1:  # Caeser
                        for q in range(len(list4)):
                            tmp = (ord(list4[q]) - 97 + 3) % 26
                            tmp = chr(tmp + 97)
                            last += tmp

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu1 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    elif (tt+1) == 2:  # Vigenere
                        for q in range(len(list4)):
                            if q % 4 == 0:
                                tmp = (ord(list4[q]) - 97 + 3) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 1:
                                tmp = (ord(list4[q]) - 97 + 4) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 2:
                                tmp = (ord(list4[q]) - 97 + 5) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            else:
                                tmp = (ord(list4[q]) - 97 + 6) % 26
                                tmp = chr(tmp + 97)
                                last += tmp

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Substitution
                        lasts = encrypt_subs(list4)

                        cnt = 0
                        for q in range(len(list5)):
                            if (lasts[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1


                if (tt+1) == 1:
                    accu1 = accu1 / len(id0)
                    accu_tmp01.append(accu1)
                    print("plain to Caeser : ", accu1)
                elif (tt+1) == 2:
                    accu2 = accu2 / len(id0)
                    accu_tmp02.append(accu2)
                    print("plain to Vigenere : ", accu2)
                else:
                    accu3 = accu3 / len(id0)
                    accu_tmp03.append(accu3)
                    print("plain to Substitution : ", accu3)


            ########################################################################
            ########################### from CT to CT ##############################
            ########################################################################

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 2 : Vigenere, 3 : Subs)
            for tt in range(4):
                if tt == 1:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                # print(len(ids))

                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 2
                accu3 = 0  # -> 3
                while e < len(id1):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id1[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id1[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Caeser
                    tmp_arr = [0 for q in range(len(list4))]

                    # decrypt to plain
                    for q in range(len(list4)):
                        tmp = (ord(list4[q]) - 97 - 3) % 26
                        tmp = chr(tmp + 97)
                        tmp_arr[q] = tmp

                    list4 = copy.deepcopy(tmp_arr)

                    if tt == 1 or tt == 0:  # Caeser or plain
                        e += 1
                        continue

                    elif (tt) == 2:  # Vigenere
                        # encrypt to Vigenere
                        for q in range(len(list4)):
                            if q % 4 == 0:
                                tmp = (ord(list4[q]) - 97 + 3) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 1:
                                tmp = (ord(list4[q]) - 97 + 4) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 2:
                                tmp = (ord(list4[q]) - 97 + 5) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            else:
                                tmp = (ord(list4[q]) - 97 + 6) % 26
                                tmp = chr(tmp + 97)
                                last += tmp

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Substitution
                        tmp_list4 = ''
                        for q in range(len(list4)):
                            tmp_list4 += list4[q]
                        lasts = encrypt_subs(tmp_list4)

                        cnt = 0
                        for q in range(len(list5)):
                            if (lasts[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1
                if tt == 1 or tt == 0:
                    continue
                elif tt == 2:
                    accu2 = accu2 / len(id1)
                    accu_CtoV.append(accu2)
                    print("Caeser to Vigenere : ", accu2)
                else:
                    accu3 = accu3 / len(id1)
                    accu_CtoS.append(accu3)
                    print("Caeser to Substitution : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Caeser, 3 : Subs)
            for tt in range(4):
                if tt == 2:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1

                # print(len(ids))

                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 1
                accu3 = 0  # -> 3
                while e < len(id2):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id2[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id2[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Vigenere
                    tmp_arr = [0 for q in range(len(list4))]

                    # decrypt to plain
                    for q in range(len(list4)):
                        if q % 4 == 0:
                            tmp = (ord(list4[q]) - 97 - 3) % 26
                            tmp = chr(tmp + 97)
                            tmp_arr[q] = tmp
                        elif q % 4 == 1:
                            tmp = (ord(list4[q]) - 97 - 4) % 26
                            tmp = chr(tmp + 97)
                            tmp_arr[q] = tmp
                        elif q % 4 == 2:
                            tmp = (ord(list4[q]) - 97 - 5) % 26
                            tmp = chr(tmp + 97)
                            tmp_arr[q] = tmp
                        else:
                            tmp = (ord(list4[q]) - 97 - 6) % 26
                            tmp = chr(tmp + 97)
                            tmp_arr[q] = tmp

                    list4 = copy.deepcopy(tmp_arr)

                    if tt == 2 or tt == 0:  # Vigenere or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Caeser
                        # encrypt to Caeser
                        for q in range(len(list4)):
                            tmp = (ord(list4[q]) - 97 + 3) % 26
                            tmp = chr(tmp + 97)
                            last += tmp

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Substitution
                        tmp_list4 = ''
                        for q in range(len(list4)):
                            tmp_list4 += list4[q]
                        lasts = encrypt_subs(tmp_list4)

                        cnt = 0
                        for q in range(len(list5)):
                            if (lasts[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1
                if tt == 2 or tt == 0:
                    continue
                elif tt == 1:
                    accu2 = accu2 / len(id2)
                    accu_VtoC.append(accu2)
                    print("Vig to Caeser : ", accu2)
                else:
                    accu3 = accu3 / len(id2)
                    accu_VtoS.append(accu3)
                    print("Vig to Substitution : ", accu3)

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # Target domain is (0 : plain, 1 : Caeser, 2 : Vigenere)
            for tt in range(4):
                if tt == 3:
                    continue

                x_fixed_fake_test = self.G(
                    x_fixed_total_test, c_fixed_list_test[tt])
                x_fixed_fake_test = self.myonehot(x_fixed_fake_test)

                iden = 0
                while (iden == 0):
                    ids = []
                    for idxstest in range(c_org_test.size(0)):
                        if (c_org_test[idxstest] == float(tt)):
                            ids.append(idxstest)
                        continue

                    if len(ids) == 0:
                        iden = 0
                        print("Fix batch again")
                        data_loader_test = self.data_loader_test

                        data_iter_test = self.data_loader_test
                        x_fixed_test, c_org_test = next(data_iter_test)
                        print("Fix batch again done")
                    else:
                        iden = 1


                e = 0
                accu1 = 0  # -> 0
                accu2 = 0  # -> 2
                accu3 = 0  # -> 3
                while e < len(id3):
                    list4 = ''  # for plain
                    list5 = ''  # for target domain tt+1

                    for q in range(100):
                        for w in range(26):
                            if (x_fixed_total_test[id3[e]][w][q].item() == 1.):
                                list4 += (chr(97+w))
                            if (x_fixed_fake_test[id3[e]][w][q].item() == 1.):
                                list5 += (chr(97+w))

                    # encrypt line
                    last = ''   # for recovered Vigenere

                    # decrypt to plain
                    list4 = decrypt(list4)

                    if tt == 3 or tt == 0:  # substitution or plain
                        e += 1
                        continue

                    elif (tt) == 1:  # Caeser
                        for q in range(len(list4)):
                            tmp = (ord(list4[q]) - 97 + 3) % 26
                            tmp = chr(tmp + 97)
                            last += tmp

                        cnt = 0
                        for q in range(len(list4)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu2 += float((len(list4) - cnt) / len(list4))
                        e += 1

                    else:  # Vigenere
                        for q in range(len(list4)):
                            if q % 4 == 0:
                                tmp = (ord(list4[q]) - 97 + 3) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 1:
                                tmp = (ord(list4[q]) - 97 + 4) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            elif q % 4 == 2:
                                tmp = (ord(list4[q]) - 97 + 5) % 26
                                tmp = chr(tmp + 97)
                                last += tmp
                            else:
                                tmp = (ord(list4[q]) - 97 + 6) % 26
                                tmp = chr(tmp + 97)
                                last += tmp

                        cnt = 0
                        for q in range(len(list5)):
                            if (last[q] != list5[q]):
                                cnt += 1
                            else:
                                continue
                        accu3 += float((len(list4) - cnt) / len(list4))
                        e += 1
                if tt == 3 or tt == 0:
                    continue

                elif tt == 1:
                    accu2 = accu2 / len(id3)
                    accu_StoC.append(accu2)
                    print("Subs to Caeser : ", accu2)
                else:
                    accu3 = accu3 / len(id3)
                    accu_StoV.append(accu3)
                    print("Subs to Vigenere : ", accu3)

            os.makedirs("accumodels/", exist_ok=True)
            AC_path = os.path.join(
                "accumodels/", "{}-ACCU-AC.ckpt".format(i))
            torch.save({'accu_tmp01': accu_tmp01, 'accu_tmp02': accu_tmp02, 'accu_tmp03': accu_tmp03,
                        'accu_CtoV': accu_CtoV, 'accu_CtoS': accu_CtoS, 'accu_VtoC': accu_VtoC,
                        'accu_VtoS': accu_VtoS, 'accu_StoC': accu_StoC, 'accu_StoV': accu_StoV}, AC_path)

