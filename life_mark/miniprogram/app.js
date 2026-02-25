// app.js - 纯本地习惯追踪，无云开发、无后端
const STORAGE_KEY = 'habit_categories';
const DEFAULT_CATEGORIES = [
  { key: 'spicy', name: '火辣辣的炎症', emoji: '🌶️' },
  { key: 'noodle', name: '超快的碳水让头发油油', emoji: '🍜' },
  { key: 'exercise', name: '暴汗燃脂时刻', emoji: '💪' },
  { key: 'social', name: '知识库贡献力', emoji: '📚' }
];

function getStoredCategories() {
  try {
    const raw = wx.getStorageSync(STORAGE_KEY);
    if (raw && Array.isArray(raw)) return raw;
  } catch (e) {}
  return null;
}

function getDefaultCategories() {
  return DEFAULT_CATEGORIES.map(cat => ({ ...cat }));
}

function getCategories() {
  const stored = getStoredCategories();
  return stored !== null && Array.isArray(stored) ? stored : getDefaultCategories();
}

function saveCategories(categories) {
  wx.setStorageSync(STORAGE_KEY, categories);
}

App({
  onLaunch() {
    // 无需云开发、无需联网，数据全部存本地
  },
  getCategories,
  saveCategories
});
