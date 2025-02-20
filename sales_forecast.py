import openpyxl
import numpy as np
import pandas as pd
from datetime import datetime
from pmdarima import auto_arima
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

def load_sales_data(file_path):
    """Загрузка данных из Excel (периоды, цены, объемы, издержки, инфляция)"""
    print(f"Загрузка данных из файла: {file_path}")
    try:
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        periods = [row.value for row in ws['A'][1:] if row.value]  # Периоды
        prices = [float(row.value) for row in ws['B'][1:] if row.value]  # Цены
        volumes = [float(row.value) for row in ws['C'][1:] if row.value]  # Объемы
        costs = [float(row.value) for row in ws['D'][1:] if row.value]  # Издержки
        # Инфляция из ячейки E1 (годовая, в процентах)
        inflation_rate = float(ws['E1'].value) / 100 if ws['E1'].value is not None else 0.05  # 5% по умолчанию
        if len(periods) != len(prices) or len(periods) != len(volumes) or len(periods) != len(costs):
            raise ValueError("Ошибка: размеры данных не совпадают")
        print(f"Успешно загружено {len(periods)} записей")
        return periods, prices, volumes, costs, inflation_rate
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        raise

def calculate_ema(series, period=3):
    """Расчет EMA"""
    return pd.Series(series).ewm(span=period, adjust=False).mean().iloc[-1]

def adjust_for_inflation(value, months, inflation_rate):
    """Корректировка значения на инфляцию"""
    monthly_inflation = (1 + inflation_rate) ** (1 / 12) - 1
    return value * (1 + monthly_inflation) ** months

def optimized_arima_forecast(series, periods, inflation_rate):
    """Оптимизированный ARIMA с учетом инфляции"""
    print(f"Прогнозирование для ряда длиной {len(series)}")
    try:
        # Отключаем сезонность, если данных меньше 12
        seasonal = False if len(series) < 12 else True
        model = auto_arima(series, start_p=0, start_q=0, max_p=3, max_q=3, d=None,
                          seasonal=seasonal, m=12 if seasonal else 1, stepwise=True,
                          suppress_warnings=True, error_action='ignore', trace=False)
        forecast = model.predict(n_periods=max(periods), return_conf_int=True)
        forecasts = {period: adjust_for_inflation(forecast[0][period-1], period, inflation_rate) 
                    for period in periods}
        conf_int = {period: (adjust_for_inflation(forecast[1][period-1][0], period, inflation_rate),
                            adjust_for_inflation(forecast[1][period-1][1], period, inflation_rate)) 
                    for period in periods}
        return forecasts, conf_int, model.order
    except Exception as e:
        print(f"Ошибка ARIMA: {str(e)}")
        return None, None, (1, 1, 0)

def calculate_volatility(series):
    """Волатильность для месячных данных"""
    log_returns = np.log(np.array(series[1:]) / np.array(series[:-1]))
    return np.std(log_returns) * np.sqrt(12)

def analyze_trend(series):
    """Анализ тренда"""
    x = np.arange(len(series))
    slope = np.polyfit(x, series, 1)[0]
    if slope > 0.01:
        return "Восходящий тренд"
    elif slope < -0.01:
        return "Нисходящий тренд"
    else:
        return "Стабильный тренд"

def calculate_correlation(prices, volumes):
    """Корреляция между ценой и объемом"""
    return np.corrcoef(prices, volumes)[0, 1]

def estimate_demand_elasticity(prices, volumes):
    """Оценка эластичности спроса"""
    log_prices = np.log(prices)
    log_volumes = np.log(volumes)
    slope = np.polyfit(log_prices, log_volumes, 1)[0]
    return slope

def demand_function(price, elasticity, base_price, base_volume):
    """Функция спроса"""
    return base_volume * (price / base_price) ** elasticity

def profit_function(price, elasticity, base_price, base_volume, cost_per_unit, months, inflation_rate):
    """Функция прибыли с учетом инфляции"""
    volume = demand_function(price, elasticity, base_price, base_volume)
    inflated_cost = adjust_for_inflation(cost_per_unit, months, inflation_rate)
    return -(price - inflated_cost) * volume

def optimize_price_volume(prices, volumes, costs, price_forecasts, volume_forecasts, periods, inflation_rate):
    """Динамическая оптимизация прибыли с учетом инфляции"""
    elasticity = estimate_demand_elasticity(prices, volumes)
    avg_cost = np.mean(costs)
    optimal_results = {}
    
    for period in periods:
        base_price = price_forecasts[period]
        base_volume = volume_forecasts[period]
        bounds = [(base_price * 0.8, base_price * 1.2)]
        
        result = minimize(profit_function, x0=[base_price], 
                         args=(elasticity, base_price, base_volume, avg_cost, period, inflation_rate),
                         bounds=bounds, method='L-BFGS-B')
        
        optimal_price = result.x[0]
        optimal_volume = demand_function(optimal_price, elasticity, base_price, base_volume)
        inflated_cost = adjust_for_inflation(avg_cost, period, inflation_rate)
        optimal_profit = (optimal_price - inflated_cost) * optimal_volume
        optimal_results[period] = (optimal_price, optimal_volume, optimal_profit)
    
    return optimal_results, elasticity, avg_cost

def process_sales_forecast(file_path):
    """Обработка данных с учетом инфляции"""
    print(f"Запуск обработки файла: {file_path}")
    try:
        periods, prices, volumes, costs, inflation_rate = load_sales_data(file_path)
        if not prices:
            print("Ошибка: данные не загружены")
            return
        
        price_trend = analyze_trend(prices)
        volume_trend = analyze_trend(volumes)
        price_volatility = calculate_volatility(prices)
        volume_volatility = calculate_volatility(volumes)
        price_ema = calculate_ema(prices)
        volume_ema = calculate_ema(volumes)
        correlation = calculate_correlation(prices, volumes)
        avg_cost = np.mean(costs)
        
        periods_forecast = [1, 3, 6, 12]
        print("Запуск прогноза для цен...")
        price_forecasts, price_conf_int, price_arima_order = optimized_arima_forecast(prices, periods_forecast, inflation_rate)
        print("Запуск прогноза для объемов...")
        volume_forecasts, volume_conf_int, volume_arima_order = optimized_arima_forecast(volumes, periods_forecast, inflation_rate)
        
        if price_forecasts is None:
            price_forecasts = {p: adjust_for_inflation(prices[-1], p, inflation_rate) for p in periods_forecast}
            price_conf_int = {p: (prices[-1]*0.95, prices[-1]*1.05) for p in periods_forecast}
            print("Использованы резервные значения для цен")
        if volume_forecasts is None:
            volume_forecasts = {p: volumes[-1] for p in periods_forecast}
            volume_conf_int = {p: (volumes[-1]*0.95, volumes[-1]*1.05) for p in periods_forecast}
            print("Использованы резервные значения для объемов")
        
        print("Запуск оптимизации...")
        optimal_results, elasticity, avg_cost = optimize_price_volume(
            prices, volumes, costs, price_forecasts, volume_forecasts, periods_forecast, inflation_rate)
        
        print(f"Открытие файла для записи: {file_path}")
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
        ws.title = "Прогноз и оптимизация"
        
        ws['E1'] = "Период (месяцы)"
        ws['F1'] = "Прогноз цены (с инфл.)"
        ws['G1'] = "Дов. интервал цены"
        ws['H1'] = "Прогноз объема"
        ws['I1'] = "Дов. интервал объема"
        ws['J1'] = "Оптимальная цена"
        ws['K1'] = "Оптимальный объем"
        ws['L1'] = "Макс. прибыль"
        ws['M1'] = "Аналитика и рекомендации"
        
        row = 2
        for period in periods_forecast:
            period_name = format_period_name(period)
            ws[f'E{row}'] = period_name
            ws[f'F{row}'] = round(price_forecasts[period], 2)
            ws[f'G{row}'] = f"[{round(price_conf_int[period][0], 2)}; {round(price_conf_int[period][1], 2)}]"
            ws[f'H{row}'] = round(volume_forecasts[period], 2)
            ws[f'I{row}'] = f"[{round(volume_conf_int[period][0], 2)}; {round(volume_conf_int[period][1], 2)}]"
            ws[f'J{row}'] = round(optimal_results[period][0], 2)
            ws[f'K{row}'] = round(optimal_results[period][1], 2)
            ws[f'L{row}'] = round(optimal_results[period][2], 2)
            row += 1
        
        ws['M2'] = f"Модель цен: ARIMA{price_arima_order}"
        ws['M3'] = f"Модель объемов: ARIMA{volume_arima_order}"
        ws['M4'] = f"Тренд цен: {price_trend}"
        ws['M5'] = f"Тренд объемов: {volume_trend}"
        ws['M6'] = f"EMA цен (3): {price_ema:.2f}"
        ws['M7'] = f"EMA объемов (3): {volume_ema:.2f}"
        ws['M8'] = f"Волатильность цен: {price_volatility:.2%}"
        ws['M9'] = f"Волатильность объемов: {volume_volatility:.2%}"
        ws['M10'] = f"Корреляция цена/объем: {correlation:.2f}"
        ws['M11'] = f"Эластичность спроса: {elasticity:.2f}"
        ws['M12'] = f"Средние издержки: {avg_cost:.2f}"
        ws['M13'] = f"Годовая инфляция: {inflation_rate*100:.2f}%"
        ws['M14'] = f"Дата расчета: {datetime.now().strftime('%Y-%m-%d')}"
        ws['M15'] = "Рекомендации:"
        ws['M16'] = ("Снизить цену для роста объема" if elasticity < -1 else 
                    "Поднять цену с учетом инфляции" if correlation > -0.5 else 
                    "Стабилизировать предложение")
        
        print(f"Сохранение изменений в файл: {file_path}")
        wb.save(file_path)
        wb.close()
        print(f"Прогноз и оптимизация успешно записаны в файл: {file_path}")
        
        print(f"Тренд цен: {price_trend}")
        print(f"Тренд объемов: {volume_trend}")
        print(f"EMA цен (3): {price_ema:.2f}")
        print(f"EMA объемов (3): {volume_ema:.2f}")
        print(f"Волатильность цен: {price_volatility:.2%}")
        print(f"Волатильность объемов: {volume_volatility:.2%}")
        print(f"Корреляция цена/объем: {correlation:.2f}")
        print(f"Эластичность спроса: {elasticity:.2f}")
        print(f"Средние издержки: {avg_cost:.2f}")
        print(f"Годовая инфляция: {inflation_rate*100:.2f}%")
        print("\nПрогноз и оптимальные значения по периодам (с учетом инфляции):")
        for period in periods_forecast:
            print(f"{format_period_name(period)}: Прогноз цены = {price_forecasts[period]:.2f} "
                  f"[{price_conf_int[period][0]:.2f}; {price_conf_int[period][1]:.2f}], "
                  f"Прогноз объема = {volume_forecasts[period]:.2f} "
                  f"[{volume_conf_int[period][0]:.2f}; {volume_conf_int[period][1]:.2f}], "
                  f"Опт. цена = {optimal_results[period][0]:.2f}, "
                  f"Опт. объем = {optimal_results[period][1]:.2f}, "
                  f"Прибыль = {optimal_results[period][2]:.2f}")
        
    except Exception as e:
        print(f"Ошибка выполнения: {str(e)}")
        raise

def format_period_name(months):
    """Форматирование периода"""
    if months == 1:
        return "1 месяц"
    elif months == 3:
        return "3 месяца"
    elif months == 6:
        return "6 месяцев"
    elif months == 12:
        return "1 год"
    return f"{months} месяцев"

def main():
    """Основная функция"""
    print("Программа прогноза и оптимизации продаж (с учетом инфляции)")
    file_path = "sales.xlsx"
    process_sales_forecast(file_path)

if __name__ == "__main__":
    main()
