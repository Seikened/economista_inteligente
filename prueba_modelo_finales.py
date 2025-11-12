import json
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score, mean_absolute_error
import matplotlib.pyplot as plt
#import seaborn as sns
import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def crear_df():
    with open("noticias_actualizadas.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    noticias = []
    for ticker, content in data.items():
        for item in content["data"]:
            item["ticker"] = ticker
            noticias.append(item)

    df = pl.DataFrame(noticias)
    
    

    
    df = (
        df.with_columns( 
            pl.col("date").str.to_datetime(strict=False)
            ).sort(["ticker","date"])
    )
    return df

def modelo_arboles(df: pl.DataFrame,features: list[str]):
    X = df.select(features).to_numpy()
    y = df["close"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    return model , mse , y_test, pred

def graficar_pred(df: pl.DataFrame, y_test, pred):
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.6)
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title("Predicción vs Real")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    #plt.show()
    

    df = df.sort("date")
    # === 5. Preparar eje temporal ===
    fechas = df["date"][-len(y_test):].to_list()  # fechas del test
    # === 6. Gráfica estilo GARCH ===
    plt.figure(figsize=(10, 5))
    plt.plot(fechas, y_test, label="Performance real", color="black", linewidth=1)
    plt.plot(fechas, pred, label="Predicción modelo", color="red", alpha=0.8)
    plt.title("Predicción vs Real (Estilo GARCH)")
    plt.xlabel("Fecha")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show(block=True)



# 'headline', 'summary', 'url', 'source', 
# 'date', 'close', 'performance', 'label', 
# 'pred_subida', 'prob_baja', 'prob_subida', 
# 'pred_subida_num', 'volatilidad_diaria', 
# 'volatilidad_anual', 'sharpe_diario', 
# 'sharpe_anual', 'sortino_diario', 'sortino_anual', 
# 'volatility', 'volatility_pred', 'ticker'


class ModeloFinanciero:
    def __init__(self,features_num, lookback=10):
        self.tensor = None
        self.features = features_num
        self.model = None
        self.X = None
        self.mask = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.mask_train = None
        self.mask_test = None
        self.lookback = lookback
    
    def crearTensor(self,JSON):
        with open(JSON,"r",encoding="utf-8") as f:
            datos = json.load(f)
        # Cargar todo el json en un DataFrame
        noticias = []
        for ticker,contenido in datos.items():
             for noticia in contenido["data"]:
                 noticia["ticker"] = ticker
                 noticias.append(noticia)
        df = pl.DataFrame(noticias)
        
        #varificar que los datos sean del tipo correcto
        df = (
            df
            .with_columns(
                pl.col("date").str.strptime(pl.Datetime,format="%Y-%m-%d %H:%M:%S%.f",strict=False)
                .dt.date()
            )
            .with_columns([
                pl.col("label").replace({"positive": 1.0, "negative": 0.0}).alias("label_num"),
                pl.col("pred_subida").replace({"positive": 1.0, "negative": 0.0}).alias("pred_subida_num")
            ])
            .with_columns(
                [pl.col(c).cast(pl.Float64) for c in self.features]
            )
            .sort(["ticker","date"])
        )
        
        '''
        df = df.with_columns(
            (pl.col("performance") - pl.col("performance").mean()) / pl.col("performance").std()
        )

        #print(df.columns)
        df = df.with_columns([
            ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
            for c in self.features
        ])
        '''
        
        # Agrupar las noticias por dia y ticker
        df = df.with_columns(
            pl.concat_list(features).alias("fila_features")
        )
        
        diario = (
            df.group_by(["ticker","date"])
                .agg(pl.col("fila_features").implode().alias("lista_noticias"),
                     pl.col("performance").mean().alias("performance"))
                .sort(["ticker","date"])
        )
        
        # rellanar dias faltantes, para que todos los dias tenga las mismas noticias 
        salida = []
        for tkr, g in diario.group_by("ticker"):
            if isinstance(tkr, (list, tuple)):  # ← para asegurarte
                tkr = tkr[0]
            # rango de fechas
            dmin = g["date"].min()
            dmax = g["date"].max()
            
            #calendario completo
            fechas = pl.select(pl.date_range(dmin, dmax, interval="1d")).to_series().to_list()
            calendario = pl.DataFrame({
                "ticker": pl.Series("ticker", [tkr] * len(fechas), dtype=pl.Utf8),
                "date": fechas
            })

            
            #unir calendario con los datos reales
            calendario = calendario.with_columns(pl.col("date").cast(pl.Date))
            g = g.with_columns(pl.col("date").cast(pl.Date))

            g_final = calendario.join(g, on=["ticker", "date"], how="left")

                        
            # llenar dias sin noticias con listas vacias
            g_final = g_final.with_columns(
                pl.when(pl.col("lista_noticias").is_null())
                    .then(pl.lit([]))
                    .otherwise(pl.col("lista_noticias"))
                    .alias("lista_noticias")
            )
            
            salida.append(g_final.sort("date"))
        diario_completo = pl.concat(salida)
        
        # Generar tensores de tamaño fijo por dia 
        max_noticias = diario_completo["lista_noticias"].list.len().max()
        
        f = len(self.features)
        
        def pad_dia(lista_noticias):
            matriz = np.zeros((max_noticias,f), dtype=np.float32)
            mascara = np.zeros((max_noticias,) , dtype=np.float32)
            for i, noticia in enumerate(lista_noticias[:max_noticias]):
                matriz[i] = noticia
                mascara[i] = 1
            return matriz, mascara
        
        X = []
        mascara = []
        
        for lista in diario_completo["lista_noticias"].to_list():
            matriz, mask = pad_dia(lista)
            X.append(matriz)
            mascara.append(mask)
            
        # Dentro de crearTensor():
        y = diario_completo["performance"].shift(-1).fill_null(0).to_numpy()
        
                # === Crear ventanas temporales ===
        lookback = self.lookback if hasattr(self, "lookback") else 5  # por defecto 5 días

        X_seq, mask_seq, y_seq = [], [], []

        for i in range(lookback, len(X)):
            X_seq.append(X[i - lookback:i])       # los últimos N días de features
            mask_seq.append(mascara[i - lookback:i])  # sus máscaras correspondientes
            y_seq.append(y[i])                    # el valor objetivo del día actual

            
        self.tensor = {
            "X":np.array(X_seq, dtype=np.float32),
            "mascara":np.array(mask_seq, dtype=np.float32),
            "y": np.array(y_seq, dtype=np.float32).reshape(-1, 1)
        }
        
        print(f"Tensor creado con forma: {self.tensor["X"].shape}\t(dias,lookback,noticias,features)")
        print(f"Mascara con forma: {self.tensor["mascara"].shape}\t\t(dia,lookback,max_noticias)")
        self.tensor_split()
        
    def tensor_split(self):
        self.X = torch.tensor(self.tensor["X"], dtype=torch.float32)
        self.mask = torch.tensor(self.tensor["mascara"], dtype=torch.float32)
        self.y = torch.tensor(self.tensor["y"], dtype=torch.float32)
        
        n = self.X.shape[0]
        i_tr = int(0.70 * n)
        i_val = int(0.85 * n)

        self.X_train, self.X_val, self.X_test = self.X[:i_tr], self.X[i_tr:i_val], self.X[i_val:]
        self.y_train, self.y_val, self.y_test = self.y[:i_tr], self.y[i_tr:i_val], self.y[i_val:]
        self.mask_train, self.mask_val, self.mask_test = self.mask[:i_tr], self.mask[i_tr:i_val], self.mask[i_val:]

        
                # Calcular media y desviación solo del conjunto de entrenamiento
        self.mu_X = self.X_train.mean(dim=(0,1,2), keepdim=True)
        self.sigma_X = self.X_train.std(dim=(0,1,2), keepdim=True).clamp(min=1e-6)


        # Normalizar X_train, X_val y X_test
        self.X_train = (self.X_train - self.mu_X) / self.sigma_X
        self.X_val   = (self.X_val - self.mu_X) / self.sigma_X
        self.X_test  = (self.X_test - self.mu_X) / self.sigma_X
        
        self.y_train_raw = self.y_train.clone()
        self.y_val_raw   = self.y_val.clone()
        self.y_test_raw  = self.y_test.clone()
        
        # Normalizar y
        self.mu_y = self.y_train.mean()
        self.sigma_y = self.y_train.std().clamp(min=1e-6)
        self.y_train = (self.y_train - self.mu_y) / self.sigma_y
        self.y_val   = (self.y_val - self.mu_y) / self.sigma_y
        self.y_test  = (self.y_test - self.mu_y) / self.sigma_y

        
        print(f"Splits -> train:{len(self.X_train)}  val:{len(self.X_val)}  test:{len(self.X_test)}")
        
    
    def entrenar_red(self, tipo="lstm", epochs=2000, lr=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # Asegura que ya corriste tensor_split()
        X_tr, y_tr, m_tr = self.X_train, self.y_train, self.mask_train
        X_val, y_val, m_val = self.X_val, self.y_val, self.mask_val

        # === Modelo (tu RedLSTM ya configurada arriba) ===
        assert self.model is not None

       # Optimizador y funciones de pérdida
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        train_loss_fn = torch.nn.L1Loss()          # robusta al ruido
        val_metric = torch.nn.MSELoss()            # se evalúa sobre regresión
        bce = torch.nn.BCEWithLogitsLoss()         # para clasificación

        alpha = 0.7                                # equilibrio entre tareas
        best_val = float("inf")
        best_state = None
        patience = 100
        wait = 0

        print(f" Entrenando con early stopping (paciencia={patience})")

        for epoch in range(epochs):
            # === TRAIN ===
            self.model.train()
            opt.zero_grad()

            pred_reg_tr, logit_tr = self.model(X_tr, m_tr)

            # pérdidas individuales
            loss_reg = train_loss_fn(pred_reg_tr, y_tr)
            y_tr_sign = (self.y_train_raw > 0).float()
            loss_cls = bce(logit_tr, y_tr_sign)

            # pérdida total
            loss_tr = alpha * loss_reg + (1 - alpha) * loss_cls
            loss_tr.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

            # === VALID ===
            self.model.eval()
            with torch.no_grad():
                pred_reg_val, logit_val = self.model(X_val, m_val)
                loss_val = val_metric(pred_reg_val, y_val).item()

                # --- Calcular accuracy direccional en validación ---
                prob_val = torch.sigmoid(logit_val)
                yv_sign = (self.y_val_raw > 0).float()

                thrs = torch.linspace(0.3, 0.7, 41)
                best_acc, best_thr = 0.0, 0.5
                for t in thrs:
                    acc = ((prob_val > t).float() == yv_sign).float().mean().item()
                    if acc > best_acc:
                        best_acc, best_thr = acc, t.item()

                self.best_thr = best_thr

            # === EARLY STOPPING ===
            if loss_val < best_val:
                best_val = loss_val
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop en epoch {epoch+1}, mejor val MSE={best_val:.4f}")
                    break

            # === Log cada 50 epochs ===
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | train {loss_tr.item():.4f} | val MSE {loss_val:.4f} | val dir-acc {best_acc*100:.2f}% | thr {best_thr:.2f}")

        # === Restaurar mejor modelo ===
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("Entrenamiento terminado.")

    
    def metricas(self):
        with torch.no_grad():
            pred_reg_ts, logit_ts = self.model(self.X_test, self.mask_test)
            y_true = self.y_test

            # MSE / MAE / R² (igual que antes)
            mse = nn.MSELoss()(pred_reg_ts, y_true).item()
            mae = torch.mean(torch.abs(pred_reg_ts - y_true)).item()
            r2 = 1 - (torch.sum((pred_reg_ts - y_true)**2) / torch.sum((y_true - y_true.mean())**2))

            # === Accuracy direccional ===
            prob_ts = torch.sigmoid(logit_ts)
            y_sign = (self.y_test_raw > 0).float()
            acc_dir = ((prob_ts > self.best_thr).float() == y_sign).float().mean().item() * 100

        print(f"\n📊 Evaluación final:")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        print(f"Accuracy direccional: {acc_dir:.2f}%")
                
        pred_np = pred_reg_ts.detach().cpu().numpy().flatten()   # ✅ corregido (usabas 'pred')
        real_np = y_true.detach().cpu().numpy().flatten()

        plt.figure(figsize=(6,6))
        plt.scatter(real_np, pred_np, alpha=0.5, edgecolors='none')
        plt.xlabel("Performance Real")
        plt.ylabel("Performance Predicho")
        plt.title("Dispersión: Real vs Predicho")
        plt.axline((0,0), slope=1, color='red', linestyle='--')
        plt.show()

        # --- Gráfica 2: Error temporal ---
        errores = real_np - pred_np
        plt.figure(figsize=(10,4))
        plt.plot(errores, label="Error", color="crimson", alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Error de predicción a lo largo del tiempo")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(8,4))
        plt.hist(prob_ts.cpu(), bins=40, color="skyblue", alpha=0.7)
        plt.axvline(self.best_thr, color="red", linestyle="--", label=f"Umbral óptimo {self.best_thr:.2f}")
        plt.title("Distribución de probabilidades de subida")
        plt.legend()
        plt.show()



     
class RedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2,dropout=0.4,bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        
        self.attention = nn.Linear(input_dim, 1)
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        # __init__ (después de definir self.fc_reg existente)
        self.fc_reg = nn.Linear(out_dim, 1)         # (si ya era self.fc, renómbrala)
        self.fc_cls = nn.Linear(out_dim, 1)         # NUEVA: para clasificación direccional

        
    def forward(self, x, mask=None):
       # x: [B, lookback, max_noticias, F]
        B, L, M, F = x.shape

        # === 1. Calcular pesos de atención por noticia ===
        attn_scores = self.attention(x)             # [B, L, M, 1]
        attn_scores = attn_scores.squeeze(-1)       # [B, L, M]

        # Aplicar máscara (si hay noticias vacías)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Normalizar pesos por softmax
        attn_weights = torch.softmax(attn_scores, dim=2)   # [B, L, M]

        # === 2. Aplicar atención (ponderar noticias) ===
        attn_weights = attn_weights.unsqueeze(-1)          # [B, L, M, 1]
        x_weighted = (x * attn_weights).sum(dim=2)         # [B, L, F]

        # === 3. Pasar la secuencia de días al LSTM ===
        out, _ = self.lstm(x_weighted)                     # [B, L, hidden*dirs]

        # === 4. Usar el último estado oculto (o promedio temporal) ===
        h_last = out.mean(dim=1)                           # [B, hidden*dirs]
        
        # forward (al final)
        y_reg = self.fc_reg(h_last)                    # regresión (magnitud)
        y_logit = self.fc_cls(h_last)                  # logits para dirección
        return y_reg, y_logit


if __name__ == "__main__":
    #df= crear_df()
    # print(df.columns)
    features=[ 'prob_baja', 'prob_subida','pred_subida_num',
       'volatility', 'volatility_pred', 'sharpe_anual', 'sortino_diario', 'sortino_anual',
       'volatilidad_diaria', 'volatilidad_anual', 'sharpe_diario','performance']
    '''
    #print(features)
    for ticker in df["ticker"].unique():
        df_t = df.filter(pl.col("ticker") == ticker)
        print(f"\n=== Entrenando modelo para {ticker} ===")
        arbol_f , mse, arbol_y, arbol_x_test = modelo_arboles(df_t, features)
        graficar_pred(df, arbol_y,arbol_x_test)

    
    #arbol_financiero , MSE_arbol_financiero = modelo_arboles(df,features)
    '''
    model = ModeloFinanciero(features) 
    model.crearTensor("noticias_actualizadas.json")
        # === Crear la red ===
    model.model = RedLSTM(
        input_dim=model.X.shape[-1],   # número de features 
        hidden_dim=128,                # puedes ajustar: 64 o 128
        output_dim=1,                 # rendimiento diario
        num_layers=2,
        dropout=0.1,                  # control del sobreajuste
        bidirectional=True
    )

    model.entrenar_red("lstm",epochs=1000)
    model.metricas()

