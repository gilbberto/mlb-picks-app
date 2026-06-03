# MLB Picks App — Historial de Sesión

## Fecha: 2 Jun 2026

### Correcciones RL
- **Bug `extract_market_odds`**: agregado `expect_point` para filtrar por -1.5 o +1.5 — evita devolver odds de la línea equivocada cuando modelo y mercado no coinciden
- **Bug `monte_carlo_predict`**: probabilidades ahora se computan por equipo/línea (`spr_home_minus`, `spr_home_plus`, `spr_away_minus`, `spr_away_plus`) en vez de siempre desde el local
- **Asignación por mercado**: RL -1.5/+1.5 ahora usa la asignación de la odds API (quien está en -1.5 en la casa) en vez del `spr_exp_margin` del modelo
- **Fallback `predict_spread`**: recalcula probs con margen firmado cuando el mercado y el modelo difieren



### Funcionalidades implementadas

- **Predicciones**: ML, RL (-1.5/+1.5), O/U con XGBoost (fallback RF)
- **Kelly Criterion**: Sizing fraccional 25%
- **Picks del Día**: Tabla con juegos, mercados, probabilidades, odds, edge
- **Registrar Picks**: Tabla + botones por juego/mercado con confirmación
- **Recomendaciones**: Top 4 picks (1 por juego, mejor edge)
- **Parlay Best Value**: Top 3 picks con EV positivo
- **Mis Picks**: Historial, bankroll chart, métricas, delete individual
- **Auto-settlement**: GitHub Actions cada 15 min (12-23 UTC)
- **Sincronización GitHub**: picks.json se sincroniza al registrar
- **Telegram**: Registro, inicio de juego, carreras, liquidación, resumen diario 8 AM
- **Mobile-responsive**: CSS dark mode con media queries

### Variables de entorno / Secrets

**Streamlit Cloud Secrets (Dashboard > Settings > Secrets):**
```toml
TELEGRAM_TOKEN = "..."
TELEGRAM_CHAT_ID = "..."
GITHUB_TOKEN = "..."
REPO = "gilbberto/mlb-picks-app"
BRANCH = "main"
```

Los valores están en `.streamlit/secrets.toml` local (gitignored) y en Streamlit Cloud Dashboard.

### Decisiones clave

- picks.json se trackea en el repo (no gitignored) para auto-settlement
- .streamlit/ está en .gitignore (secrets locales)
- ZoneInfo "America/Chihuahua" para timezone
- XGBoost es el modelo primario, RF es fallback
- La línea O/U se muestra (ej. "Over 8.5") en todos lados
- Solo Best Value parlay (Top Picks y High Confidence eliminados)
- 3 niveles de confianza: 🔥🔥🔥 >8%, 🔥🔥 >5%, 🔥 >2%

### Archivos principales

| Archivo | Propósito |
|---------|-----------|
| app.py | Streamlit app principal |
| bankroll.py | Kelly, P&L, auto-settlement, calibración |
| settle_and_notify.py | GitHub Actions: settlement + Telegram |
| morning_summary.py | Resumen diario 8 AM |
| xgb_hw/rd/tot.pkl | Modelos XGBoost entrenados |
| rf_hw/rd/tot.pkl | Modelos Random Forest (fallback) |
| picks.json | Base de datos de picks |
| game_starts_notified.json | Tracking de juegos notificados |
| .github/workflows/auto-settle.yml | Settlement cada 15 min |
| .github/workflows/morning-summary.yml | Resumen diario 8 AM |
| .streamlit/secrets.toml | Secrets locales (gitignored) |

### Fixes 2 Jun 2026
- **Notificaciones**: ahora solo envía inicio/carreras de juegos con picks registrados (antes mandaba todos)
- **Win probability en carreras**: cada notificación de carrera incluye % de ganar de cada pick registrado en ese juego
- **market_fav_team** movido después de `og = match_game()` (antes usaba og del juego anterior)
- **Fallback path**: recalcula probs con margen firmado cuando mercado y modelo difieren
- **auto-settle.yml**: cambiado `git pull --rebase` por `fetch + rebase` para detached HEAD
- **auto-settle.yml**: corregido YAML duplicado que causaba "No jobs were run"

### Pendientes / Ideas

- Dashboard de rendimiento semanal/mensual
- Notas en picks (comentarios)
- Win rate por mercado (Ya en Mis Picks)
- Calibración del modelo (Ya en Mis Picks)
