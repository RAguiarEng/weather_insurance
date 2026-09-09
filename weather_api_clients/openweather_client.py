"""Cliente HTTP resiliente para a API OpenWeatherMap
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

import requests
from typing import Dict, Optional, Tuple, Any
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    OPENWEATHERMAP_API_KEY,
    OPENWEATHERMAP_BASE_URL,
    OPENWEATHERMAP_FORECAST_URL,
    OPENWEATHERMAP_DEFAULT_LANG,
    OPENWEATHERMAP_DEFAULT_UNITS,
    OPENWEATHERMAP_TIMEOUT_SECONDS,
    OPENWEATHERMAP_MAX_RETRIES
)

class OpenWeatherMapClient:
    def __init__(self):
        self.api_key = OPENWEATHERMAP_API_KEY
        if not self.api_key:
            logger.warning("Chave OPENWEATHERMAP_API_KEY não encontrada no ambiente.")
            
        self.base_url = OPENWEATHERMAP_BASE_URL
        self.forecast_url = OPENWEATHERMAP_FORECAST_URL
        self.default_params = {
            'appid': self.api_key,
            'lang': OPENWEATHERMAP_DEFAULT_LANG,
            'units': OPENWEATHERMAP_DEFAULT_UNITS
        }
        self.session = requests.Session()
        
    @retry(
        stop=stop_after_attempt(OPENWEATHERMAP_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def get_current_weather(
        self,
        lat: float,
        lon: float,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Obtém dados meteorológicos em tempo real para as coordenadas."""
        params = {**self.default_params, 'lat': lat, 'lon': lon}
        if extra_params:
            params.update(extra_params)
            
        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=OPENWEATHERMAP_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na requisição OpenWeatherMap (Current): {str(e)}")
            return False, None

    @retry(
        stop=stop_after_attempt(OPENWEATHERMAP_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(requests.exceptions.RequestException)
    )
    def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 5,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Obtém previsão do tempo (forecast) para identificar riscos antecipadamente."""
        params = {**self.default_params, 'lat': lat, 'lon': lon}
        if extra_params:
            params.update(extra_params)
            
        try:
            response = self.session.get(
                self.forecast_url,
                params=params,
                timeout=OPENWEATHERMAP_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na requisição OpenWeatherMap (Forecast): {str(e)}")
            return False, None

    def normalize_current_weather(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza o payload JSON bruto da OpenWeatherMap em um formato padronizado."""
        if not raw_data:
            return {}
        
        main = raw_data.get("main", {})
        wind = raw_data.get("wind", {})
        weather = raw_data.get("weather", [{}])[0]
        rain = raw_data.get("rain", {})

        return {
            "city": raw_data.get("name", "Desconhecido"),
            "temp": main.get("temp", 0.0),
            "feels_like": main.get("feels_like", 0.0),
            "temp_min": main.get("temp_min", 0.0),
            "temp_max": main.get("temp_max", 0.0),
            "humidity": main.get("humidity", 0),
            "pressure": main.get("pressure", 0),
            "wind_speed_kmh": round(wind.get("speed", 0.0) * 3.6, 1), # m/s para km/h
            "wind_gust_kmh": round(wind.get("gust", 0.0) * 3.6, 1) if "gust" in wind else None,
            "rain_1h_mm": rain.get("1h", 0.0),
            "condition_main": weather.get("main", ""),
            "condition_description": weather.get("description", ""),
            "icon": weather.get("icon", "")
        }
